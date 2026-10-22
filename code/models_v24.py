"""
v24: Multi-scale dilated Conv + BiLSTM + Attention + Subject Embedding.

v23에서 Transformer + MaxPool이 역효과임을 확인.
→ Transformer 제거, MaxPool 제거, BiLSTM 복귀.
→ multi-scale conv 블록만 v8 대비 변경점으로 단독 효과 검증.

v8 대비 변경:
  before: Conv1d(24→64, k=3) + Conv1d(64→128, k=3)
  after:  multi-scale(k=3/15/51, d=1/2/4) → concat → Conv1d(1×1) → 128ch

수용 영역:
  k=3,  d=1: RF = 3   timestep =  1.5ms
  k=15, d=2: RF = 29  timestep = 14.5ms
  k=51, d=4: RF = 201 timestep = 100.5ms
"""
import torch
import torch.nn as nn


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.b1 = self._branch(in_ch, out_ch, k=3,  dilation=1)
        self.b2 = self._branch(in_ch, out_ch, k=15, dilation=2)
        self.b3 = self._branch(in_ch, out_ch, k=51, dilation=4)
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


class MultiScaleBiLSTMModel(nn.Module):
    """
    Input : (B, C=24, T=600)  channels-first
    Output: (B, n_classes)
    """
    def __init__(self, n_channels: int, n_classes: int, n_subjects: int):
        super().__init__()
        # (B, 24, 600) → (B, 128, 600)
        self.ms_conv = MultiScaleConvBlock(in_ch=n_channels, out_ch=64)

        # (B, 600, 128) → (B, 600, 256)  — v8과 동일 hidden=128, 2layer, bidir
        self.lstm = nn.LSTM(
            128, 128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.attn     = nn.Linear(256, 1)
        self.subj_emb = nn.Embedding(n_subjects, 32)
        self.head     = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 + 32, n_classes),   # 288 → 50
        )

    def forward(self, x, s):
        h = self.ms_conv(x)          # (B, 128, 600)
        h = h.transpose(1, 2)        # (B, 600, 128)
        h, _ = self.lstm(h)          # (B, 600, 256)
        a = torch.softmax(self.attn(h), dim=1)
        feat = (h * a).sum(dim=1)    # (B, 256)
        emb  = self.subj_emb(s)      # (B, 32)
        return self.head(torch.cat([feat, emb], dim=1))
