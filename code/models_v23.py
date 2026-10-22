"""
v23: Multi-scale dilated Conv + MaxPool + Transformer encoder + Subject Embedding.

MiniMax-M3 LLM 추천 아키텍처.

기존 v8 병목:
  - Conv1d(k=3) = 1.5ms 컨텍스트만 봄 (EMG 패턴은 50~200ms 스케일)
  - 600 timestep LSTM 비효율 (대부분 정보는 20Hz 이하)

개선:
  - multi-scale dilated conv: k=3/15/51, dilation=1/2/4
    → 1.5ms / 14.5ms / 100.5ms 수용 영역을 병렬로 커버
  - MaxPool(stride=4): 600→150 다운샘플링 (LSTM 제거)
  - TransformerEncoder: 2L, d=128, 4head, batch_first=True
  - subject embedding(32) 유지
"""
import torch
import torch.nn as nn
import math


class MultiScaleConvBlock(nn.Module):
    """세 가지 시간 스케일을 병렬로 커버하는 conv 블록."""
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        # 각 branch의 same-padding: dilation * (k-1) // 2
        self.b1 = self._branch(in_ch, out_ch, k=3,  dilation=1)   # 1.5ms RF
        self.b2 = self._branch(in_ch, out_ch, k=15, dilation=2)   # 14.5ms RF
        self.b3 = self._branch(in_ch, out_ch, k=51, dilation=4)   # 100.5ms RF
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch * 3, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    @staticmethod
    def _branch(in_ch, out_ch, k, dilation):
        pad = dilation * (k - 1) // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.proj(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiScaleTransformerEMG(nn.Module):
    """
    Input  : (B, C=24, T=600)  channels-first
    Output : (B, n_classes)
    """
    def __init__(self, n_channels: int, n_classes: int, n_subjects: int):
        super().__init__()

        # 1) multi-scale conv: (B, 24, 600) → (B, 128, 600)
        self.ms_conv = MultiScaleConvBlock(in_ch=n_channels, out_ch=64)

        # 2) 다운샘플: (B, 128, 600) → (B, 128, 150)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        # 3) Positional encoding + Transformer
        self.pos_enc = PositionalEncoding(d_model=128, max_len=200, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.1, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4) Attention-weighted pooling: (B, 150, 128) → (B, 128)
        self.attn = nn.Linear(128, 1)

        # 5) Subject embedding + classifier
        self.subj_emb = nn.Embedding(n_subjects, 32)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 + 32, n_classes),   # 160 → 50
        )

    def forward(self, x, s):
        # x: (B, C, T), s: (B,) 0-indexed subject id
        h = self.ms_conv(x)          # (B, 128, 600)
        h = self.pool(h)             # (B, 128, 150)
        h = h.transpose(1, 2)        # (B, 150, 128)
        h = self.pos_enc(h)
        h = self.transformer(h)      # (B, 150, 128)

        a = torch.softmax(self.attn(h), dim=1)   # (B, 150, 1)
        feat = (h * a).sum(dim=1)                # (B, 128)

        emb = self.subj_emb(s)                   # (B, 32)
        return self.head(torch.cat([feat, emb], dim=1))
