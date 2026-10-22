"""
v54: 스펙트로그램 2D-CNN — 근본적으로 다른 아키텍처(앙상블 다양성용).

동기: 기존 앙상블 멤버는 전부 1D-CNN-LSTM 계열 → 오류가 상관됨. 시간-주파수
      스펙트로그램을 2D-CNN으로 처리하면 완전히 다른 귀납편향 → 비상관 오류
      → 앙상블에 새 다양성. 단독 성능이 다소 낮아도 앙상블 기여 기대(v47 교훈).

입력: 전처리전달_v28의 24ch 시간영역 윈도우(600샘플) — 기존 멤버와 동일.
      모델 내부에서 채널별 STFT → 로그-크기 스펙트로그램 (24, F, T) → 2D-CNN.
출력: 50클래스 (동일 확률공간) + subject embedding. → 앙상블 drop-in 멤버.
backbone 이름 'cnn' 유지(관례).
"""
import torch
import torch.nn as nn

N_FFT = 64
HOP   = 16


def compute_spectrogram(x, n_fft=N_FFT, hop=HOP):
    """x: (B, C, T) 시간영역 → (B, C, F, Tf) 로그-크기 스펙트로그램."""
    B, C, T = x.shape
    xr = x.reshape(B * C, T)
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    st = torch.stft(xr, n_fft=n_fft, hop_length=hop, window=win,
                    center=True, return_complex=True)     # (B*C, F, Tf)
    mag = torch.log1p(st.abs())
    Fb, Tf = mag.shape[1], mag.shape[2]
    return mag.reshape(B, C, Fb, Tf)


class SpectrogramCNN(nn.Module):
    def __init__(self, n_channels=24, n_classes=50, n_subjects=40, emb_dim=32):
        super().__init__()
        self.subject_emb = nn.Embedding(n_subjects + 1, emb_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.BatchNorm2d(64), nn.ELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),         nn.BatchNorm2d(128), nn.ELU(),
            nn.AdaptiveAvgPool2d(1),                  # (B,128,1,1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128 + emb_dim, n_classes)
        )

    def forward(self, x, subject_idx):
        spec = compute_spectrogram(x)               # (B, 24, F, T)
        h = self.cnn(spec).flatten(1)               # (B, 128)
        s = self.subject_emb(subject_idx)           # (B, 32)
        return self.fc(torch.cat([h, s], dim=1))    # (B, n_classes)
