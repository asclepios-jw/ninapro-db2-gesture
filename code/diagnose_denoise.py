"""
v13 1단계: 웨이블릿 디노이징이 SNR/separability를 실제로 끌어올리는지 진단
(모델 재학습 없이, v12와 동일한 방식으로 40명 전체 비교).
특히 v12에서 확인된 "어려운 5명"(S28,S31,S04,S26,S07)에게 더 큰 효과가
있는지를 본다.
"""
import time
import json
import numpy as np
from multiprocessing import Pool
from scipy.stats import f_oneway, pearsonr

import data_utils as du

SUBJECTS = du.SUBJECTS
WIN_MS, STEP_MS, MIN_PURITY = 300, 150, 0.8
HARD_SUBJECTS = {28, 31, 4, 26, 7}


def _compute_metrics(emg, labels, reps):
    rest_mask = labels == 0
    move_mask = ~rest_mask
    rms_rest = np.sqrt(np.mean(emg[rest_mask] ** 2, axis=0))
    rms_move = np.sqrt(np.mean(emg[move_mask] ** 2, axis=0))
    snr_db = 20 * np.log10(np.mean(rms_move) / (np.mean(rms_rest) + 1e-12))

    win_len = int(WIN_MS / 1000 * du.FS)
    step_len = int(STEP_MS / 1000 * du.FS)
    X, y, r = du.segment_signal(emg, labels, reps, win_len=win_len, step_len=step_len, min_purity=MIN_PURITY)
    train_mask = ~np.isin(r, du.TEST_REPS + du.VAL_REPS)
    X_train, y_train = X[train_mask], y[train_mask]

    mav = np.mean(np.abs(X_train), axis=1)
    f_scores = []
    for ch in range(mav.shape[1]):
        groups = [mav[y_train == c, ch] for c in np.unique(y_train)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, _ = f_oneway(*groups)
        if np.isfinite(f_stat):
            f_scores.append(f_stat)
    sep = float(np.mean(f_scores)) if f_scores else float("nan")
    return float(snr_db), sep


def _worker(subject_id):
    t0 = time.time()
    raw = du.load_subject_raw(subject_id)
    emg = du.preprocess_emg(raw["emg"])
    snr_before, sep_before = _compute_metrics(emg, raw["labels"], raw["reps"])

    emg_dn = du.wavelet_denoise(emg)
    snr_after, sep_after = _compute_metrics(emg_dn, raw["labels"], raw["reps"])

    elapsed = time.time() - t0
    tag = "[HARD]" if subject_id in HARD_SUBJECTS else ""
    print(f"[S{subject_id:02d}]{tag} SNR {snr_before:.2f}->{snr_after:.2f}dB "
          f"({snr_after-snr_before:+.2f}) | sep {sep_before:.1f}->{sep_after:.1f} "
          f"({sep_after-sep_before:+.1f}) | {elapsed:.1f}s", flush=True)

    return {
        "subject": subject_id, "is_hard": subject_id in HARD_SUBJECTS,
        "snr_before": snr_before, "snr_after": snr_after,
        "sep_before": sep_before, "sep_after": sep_after,
    }


def main():
    with Pool(8) as pool:
        results = pool.map(_worker, SUBJECTS)

    with open("../results/v13/diagnose_denoise.json", "w") as f:
        json.dump(results, f, indent=2)

    hard = [r for r in results if r["is_hard"]]
    easy = [r for r in results if not r["is_hard"]]
    print("\n=== 요약 ===")
    print(f"전체 평균 SNR 변화: {np.mean([r['snr_after']-r['snr_before'] for r in results]):+.3f} dB")
    print(f"어려운 5명 평균 SNR 변화: {np.mean([r['snr_after']-r['snr_before'] for r in hard]):+.3f} dB")
    print(f"나머지 35명 평균 SNR 변화: {np.mean([r['snr_after']-r['snr_before'] for r in easy]):+.3f} dB")
    print(f"전체 평균 separability 변화: {np.mean([r['sep_after']-r['sep_before'] for r in results]):+.2f}")
    print(f"어려운 5명 평균 separability 변화: {np.mean([r['sep_after']-r['sep_before'] for r in hard]):+.2f}")
    print(f"나머지 35명 평균 separability 변화: {np.mean([r['sep_after']-r['sep_before'] for r in easy]):+.2f}")


if __name__ == "__main__":
    main()
