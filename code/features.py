"""
EMG 윈도우 (N, T, C) -> 특징 벡터 (N, n_features) 추출.
시간 도메인 (MAV, RMS, WL, ZC, SSC) + 주파수 도메인 (MNF, MDF, PKF), 채널별 계산.
대규모(N~백만) 처리를 위해 numpy 벡터화 + welch 배치 처리 사용.
"""
import numpy as np
from scipy.signal import welch

FS = 2000
ZC_THRESH = 1e-4
SSC_THRESH = 1e-4


def _time_domain_features(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C) -> (N, C, 5) [MAV, RMS, WL, ZC, SSC]"""
    N, T, C = X.shape
    mav = np.mean(np.abs(X), axis=1)                       # (N, C)
    rms = np.sqrt(np.mean(X ** 2, axis=1))                 # (N, C)
    diff1 = np.diff(X, axis=1)                              # (N, T-1, C)
    wl = np.sum(np.abs(diff1), axis=1)                      # (N, C)
    zc = np.sum((np.abs(diff1) > ZC_THRESH) &
                (np.sign(X[:, :-1, :]) != np.sign(X[:, 1:, :])), axis=1).astype(np.float32)
    diff2 = np.diff(diff1, axis=1)                          # (N, T-2, C)
    ssc = np.sum((diff1[:, :-1, :] * diff1[:, 1:, :] < 0) &
                 (np.abs(diff2) > SSC_THRESH), axis=1).astype(np.float32)
    return np.stack([mav, rms, wl, zc, ssc], axis=-1)       # (N, C, 5)


def _freq_domain_features(X: np.ndarray, fs: int = FS, batch_size: int = 4000) -> np.ndarray:
    """X: (N, T, C) -> (N, C, 3) [MNF, MDF, PKF], batched welch over (batch*C, T)."""
    N, T, C = X.shape
    out = np.empty((N, C, 3), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = X[start:end].transpose(0, 2, 1).reshape(-1, T)  # (batch*C, T)
        f, Pxx = welch(chunk, fs=fs, nperseg=min(256, T), axis=-1)
        Pxx_sum = np.sum(Pxx, axis=-1, keepdims=True) + 1e-12
        mnf = np.sum(f[None, :] * Pxx, axis=-1) / Pxx_sum.squeeze(-1)
        cumpow = np.cumsum(Pxx, axis=-1)
        half = Pxx_sum.squeeze(-1) / 2
        mdf_idx = np.argmax(cumpow >= half[:, None], axis=-1)
        mdf = f[mdf_idx]
        pkf_idx = np.argmax(Pxx, axis=-1)
        pkf = f[pkf_idx]
        feats = np.stack([mnf, mdf, pkf], axis=-1).reshape(end - start, C, 3)
        out[start:end] = feats

    return out


def extract_features(X: np.ndarray, fs: int = FS) -> np.ndarray:
    """X: (N, T, C) -> (N, C*8) feature matrix."""
    td = _time_domain_features(X)            # (N, C, 5)
    fd = _freq_domain_features(X, fs=fs)      # (N, C, 3)
    feats = np.concatenate([td, fd], axis=-1)  # (N, C, 8)
    N, C, F = feats.shape
    return feats.reshape(N, C * F).astype(np.float32)
