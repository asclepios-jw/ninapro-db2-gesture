"""
v12: 왜 어떤 피험자(S28, S31 등)는 항상 어렵고 어떤 피험자(S01, S05, S40)는
항상 쉬운지 신호 자체에서 원인을 찾는 진단 분석. 모델을 다시 학습하지 않고,
원본 필터링 신호(피험자별 정규화 적용 전)에서 직접 계산 가능한 지표들을
40명 전체에 대해 뽑아서 v9의 base_acc(피험자별 풀링 모델 정확도)와 상관관계를 본다.

계산하는 지표:
  1. SNR(dB)            : 움직임 구간 RMS 대비 rest 구간 RMS (피부 임피던스, 전극
                          부착 품질 등에 따라 달라질 수 있음)
  2. channel_cv         : 채널별 RMS의 변동계수(CV) — 특정 채널만 비정상적으로
                          크거나 작으면(전극 불량 등) 이 값이 커짐
  3. separability_score : 50개 클래스 간 MAV 특징의 분리도 (One-way ANOVA F값,
                          채널 평균) — 클수록 클래스들이 신호상으로 더 잘 구분됨
"""
import time
import json
import numpy as np
from multiprocessing import Pool
from scipy.stats import f_oneway

import data_utils as du

SUBJECTS = du.SUBJECTS
WIN_MS, STEP_MS, MIN_PURITY = 300, 150, 0.8


def _diagnose_one(subject_id):
    t0 = time.time()
    raw = du.load_subject_raw(subject_id)
    emg = du.preprocess_emg(raw["emg"])  # 대역통과+노치만, 피험자별 정규화 X
    labels, reps = raw["labels"], raw["reps"]

    rest_mask = labels == 0
    move_mask = ~rest_mask

    rms_rest = np.sqrt(np.mean(emg[rest_mask] ** 2, axis=0))      # (12,)
    rms_move = np.sqrt(np.mean(emg[move_mask] ** 2, axis=0))      # (12,)
    snr_db = 20 * np.log10(np.mean(rms_move) / (np.mean(rms_rest) + 1e-12))
    channel_cv = float(np.std(rms_move) / (np.mean(rms_move) + 1e-12))

    win_len = int(WIN_MS / 1000 * du.FS)
    step_len = int(STEP_MS / 1000 * du.FS)
    X, y, r = du.segment_signal(emg, labels, reps, win_len=win_len, step_len=step_len, min_purity=MIN_PURITY)
    train_mask = ~np.isin(r, du.TEST_REPS + du.VAL_REPS)
    X_train, y_train = X[train_mask], y[train_mask]

    mav = np.mean(np.abs(X_train), axis=1)  # (N, 12)
    f_scores = []
    for ch in range(mav.shape[1]):
        groups = [mav[y_train == c, ch] for c in np.unique(y_train)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, _ = f_oneway(*groups)
        if np.isfinite(f_stat):
            f_scores.append(f_stat)
    separability_score = float(np.mean(f_scores)) if f_scores else float("nan")

    elapsed = time.time() - t0
    print(f"[S{subject_id:02d}] SNR={snr_db:.2f}dB ch_cv={channel_cv:.3f} "
          f"separability={separability_score:.1f} ({elapsed:.1f}s)", flush=True)

    return {
        "subject": subject_id, "snr_db": float(snr_db), "channel_cv": channel_cv,
        "separability_score": separability_score, "rms_rest_mean": float(np.mean(rms_rest)),
        "rms_move_mean": float(np.mean(rms_move)),
    }


def main():
    with Pool(8) as pool:
        results = pool.map(_diagnose_one, SUBJECTS)

    with open("diagnose_subjects.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved diagnose_subjects.json")


if __name__ == "__main__":
    main()
