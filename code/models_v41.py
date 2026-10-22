"""
v41: 깊은 Residual CNN backbone + v8 헤드(subject emb→LSTM→Attention).

v8/v40 backbone(2-conv)을 residual block 기반으로 심화한다.
헤드(LSTM + subject embedding + Attention + fc)와 forward 구조는 v8과 100% 동일.
backbone 모듈 이름은 'cnn' 유지 → finetune의 freeze_cnn 로직 그대로 작동.

backbone 구조 (입력 (B,24,600)):
  Stem : Conv k5 → BN → ELU → MaxPool(2)          600→300, 24→64
  ResBlock(64→64)  x2                              300 (skip)
  MaxPool(2)                                       300→150
  ResBlock(64→128) + ResBlock(128→128)            150
  MaxPool(2)                                       150→75
  ResBlock(128→256)                               75
  출력 (B, 256, 75)  → 헤드로

v8 backbone 대비:
  - 채널 128 → 256 (표현력↑)
  - 깊이 2-conv → stem + 5 residual block (skip connection으로 학습 안정)
  - 시퀀스 150 → 75 (LSTM 입력 절반, 더 높은 추상도 + 빠름)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """v8과 동일 (tanh + Linear 비선형 attention)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights


class ResBlock1d(nn.Module):
    """1D residual basic block (post-activation, ELU). 채널 변화 시 1x1 proj skip."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.ELU()
        self.proj  = (nn.Conv1d(in_ch, out_ch, kernel_size=1)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        identity = self.proj(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + identity)


class SubjectAwareResCNNLSTMAttention(nn.Module):
    def __init__(self, n_channels=24, n_classes=50, n_subjects=40,
                 emb_dim=32, lstm_hidden=128, lstm_layers=2, cnn_out=256):
        super().__init__()
        self.subject_emb = nn.Embedding(n_subjects + 1, emb_dim)

        # backbone (이름 'cnn' 유지 — freeze_cnn 호환)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),   # 600→300
            ResBlock1d(64, 64),
            ResBlock1d(64, 64),
            nn.MaxPool1d(2),                                  # 300→150
            ResBlock1d(64, 128),
            ResBlock1d(128, 128),
            nn.MaxPool1d(2),                                  # 150→75
            ResBlock1d(128, cnn_out),                         # →256
        )

        self.lstm = nn.LSTM(cnn_out + emb_dim, lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden * 2)
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(lstm_hidden * 2, n_classes)
        )

    def forward(self, x, subject_idx):
        s_emb = self.subject_emb(subject_idx)

        h = self.cnn(x)                 # (B, cnn_out, T')
        h = h.transpose(1, 2)           # (B, T', cnn_out)

        T_prime = h.shape[1]
        s_emb_expanded = s_emb.unsqueeze(1).expand(-1, T_prime, -1)
        h_concat = torch.cat([h, s_emb_expanded], dim=-1)

        out, _ = self.lstm(h_concat)
        context, _ = self.attention(out)

        return self.fc(context)
