"""
v45: v44(TD 특징) + 주파수영역 특징(FD feature) 추가.

동기: v44에서 시간영역 특징(RMS/MAV/WL/ZC/SSC)이 hard를 균일하게 개선(+0.93p,
      재현 검증됨). 주파수영역 특징은 근피로·발화율(스펙트럼 분포)을 담아
      시간영역과 상보적 → 이 방향을 심화.

FD 특징 (12 EMG 채널 각각, FFT 기반):
  MNF (mean frequency)   : Σ(f·P)/ΣP — 스펙트럼 무게중심
  MDF (median frequency) : 누적 파워가 절반이 되는 주파수 — 피로 지표
  PKF (peak frequency)   : 최대 파워 주파수
  = 3 × 12 = 36. 정규화에 불변(비율/argmax) → 안전.

특징 총합: TD 60 + FD 36 = 96 → feat_mlp(BatchNorm→MLP) → context와 융합.
헤드/forward 골격은 v44와 동일. backbone 이름 'cnn' 유지(freeze_cnn 호환).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# v44에서 재사용
from models_v44 import Attention, compute_td_features

FS = 2000  # 샘플링 주파수


def compute_fd_features(emg, fs=FS, eps=1e-8):
    """주파수영역 특징 3종 × 채널. emg: (B, Ce, T) → (B, 3*Ce). 정규화 불변."""
    B, Ce, T = emg.shape
    X = torch.fft.rfft(emg, dim=2)                 # (B, Ce, Fbin)
    P = X.real ** 2 + X.imag ** 2                  # 파워 스펙트럼 (B, Ce, Fbin)
    Fbin = P.shape[2]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(emg.device)  # (Fbin,)

    Ptot = P.sum(dim=2) + eps                      # (B, Ce)
    mnf = (P * freqs.view(1, 1, Fbin)).sum(dim=2) / Ptot      # (B, Ce) 평균주파수
    # MDF: 누적 파워가 절반 이상이 되는 첫 주파수 (누적은 단조 → argmax가 첫 지점)
    cum = torch.cumsum(P, dim=2)
    reached = (cum >= (0.5 * Ptot).unsqueeze(2)).float()
    mdf = freqs[torch.argmax(reached, dim=2)]      # (B, Ce) 중앙주파수
    pkf = freqs[torch.argmax(P, dim=2)]            # (B, Ce) 피크주파수
    return torch.cat([mnf, mdf, pkf], dim=1)       # (B, 3*Ce)


class SubjectAwareCNNLSTMAttentionFeatFD(nn.Module):
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

        n_feat = 5 * n_emg + 3 * n_emg           # TD 60 + FD 36 = 96
        self.feat_mlp = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Linear(n_feat, feat_hidden), nn.BatchNorm1d(feat_hidden), nn.ELU(),
            nn.Linear(feat_hidden, feat_hidden), nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(lstm_hidden * 2 + feat_hidden, n_classes)
        )

    def forward(self, x, subject_idx):
        emg = x[:, :self.n_emg, :]
        feats = torch.cat([compute_td_features(emg), compute_fd_features(emg)], dim=1)  # (B, 96)

        s_emb = self.subject_emb(subject_idx)
        h = self.cnn(x)
        h = h.transpose(1, 2)
        T_prime = h.shape[1]
        s_emb_exp = s_emb.unsqueeze(1).expand(-1, T_prime, -1)
        h = torch.cat([h, s_emb_exp], dim=-1)
        out, _ = self.lstm(h)
        context, _ = self.attention(out)

        f = self.feat_mlp(feats)
        return self.fc(torch.cat([context, f], dim=1))
