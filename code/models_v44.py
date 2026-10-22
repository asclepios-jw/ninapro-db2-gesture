"""
v44: v8 아키텍처 + 수공 시간영역 특징(handcrafted TD feature) 병렬 브랜치.

동기: v41에서 backbone 용량은 병목이 아님을, v43에서 전극 shift 증강은 실패를
      확인. 남은 방향 = 모델에 주는 '정보'를 바꾸기.
      Ninapro DB2에서 오래 검증된 시간영역 특징(Hudgins set + WL)을
      학습된 CNN 특징과 융합해 보완한다.

특징 (12 EMG 채널 각각, ch 0–11):
  RMS, MAV, WL(waveform length), ZC(zero crossings), SSC(slope sign changes)
  → 5 × 12 = 60 특징. 입력 BatchNorm으로 스케일 차이 정규화 후 MLP.

구조:
  [CNN→subjEmb→LSTM→Attention] context(256)  ┐
                                              ├ concat → Dropout → Linear(320→50)
  [TD feature 60 → MLP → 64]                  ┘

헤드/forward는 v8과 동일 골격 + feat 브랜치. backbone 이름 'cnn' 유지
→ finetune freeze_cnn 그대로 작동 (feat_mlp·fc는 학습 대상).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """v8과 동일."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_out):
        w = F.softmax(self.attention(lstm_out), dim=1)
        return torch.sum(w * lstm_out, dim=1), w


def compute_td_features(emg, thr=0.01, eps=1e-6):
    """시간영역 특징 5종 × 채널. emg: (B, Ce, T) → (B, 5*Ce).
    입력은 정규화된 필터 EMG(std≈0.9)이므로 thr은 작은 값."""
    rms = torch.sqrt(torch.mean(emg * emg, dim=2) + eps)          # (B, Ce)
    mav = torch.mean(emg.abs(), dim=2)                             # (B, Ce)
    d   = emg[:, :, 1:] - emg[:, :, :-1]
    wl  = d.abs().sum(dim=2)                                       # (B, Ce)
    # ZC: 부호 변화 + 진폭차 임계
    prod = emg[:, :, :-1] * emg[:, :, 1:]
    zc   = (((prod < 0) & (d.abs() > thr)).float()).sum(dim=2)     # (B, Ce)
    # SSC: 기울기 부호 변화 + 크기 임계
    d1  = emg[:, :, 1:-1] - emg[:, :, :-2]
    d2  = emg[:, :, 2:]   - emg[:, :, 1:-1]
    ssc = (((d1 * d2) < 0) & ((d1.abs() > thr) | (d2.abs() > thr))).float().sum(dim=2)
    return torch.cat([rms, mav, wl, zc, ssc], dim=1)              # (B, 5*Ce)


class SubjectAwareCNNLSTMAttentionFeat(nn.Module):
    def __init__(self, n_channels=24, n_classes=50, n_subjects=40, emb_dim=32,
                 lstm_hidden=128, lstm_layers=2, n_emg=12, feat_hidden=64):
        super().__init__()
        self.n_emg = n_emg
        self.subject_emb = nn.Embedding(n_subjects + 1, emb_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ELU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128 + emb_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden * 2)

        n_feat = 5 * n_emg
        self.feat_mlp = nn.Sequential(
            nn.BatchNorm1d(n_feat),                      # 특징 스케일 정규화
            nn.Linear(n_feat, feat_hidden), nn.BatchNorm1d(feat_hidden), nn.ELU(),
            nn.Linear(feat_hidden, feat_hidden), nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(lstm_hidden * 2 + feat_hidden, n_classes)
        )

    def forward(self, x, subject_idx):
        feats = compute_td_features(x[:, :self.n_emg, :])     # (B, 60) EMG 채널에서

        s_emb = self.subject_emb(subject_idx)
        h = self.cnn(x)
        h = h.transpose(1, 2)
        T_prime = h.shape[1]
        s_emb_exp = s_emb.unsqueeze(1).expand(-1, T_prime, -1)
        h = torch.cat([h, s_emb_exp], dim=-1)
        out, _ = self.lstm(h)
        context, _ = self.attention(out)                     # (B, 256)

        f = self.feat_mlp(feats)                             # (B, 64)
        return self.fc(torch.cat([context, f], dim=1))       # (B, n_classes)
