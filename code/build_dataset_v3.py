"""
v8 2단계: 스윕에서 고른 최적 설정(윈도우 300ms/스텝 150ms, 라벨 순도>=0.8)으로
40명 피험자 전체를 재구축. 피험자별 정규화는 v2와 동일하게 유지.

출력 위치: ninapro_db2_project/data_processed/v3/
"""
import os
import time
import numpy as np
from multiprocessing import Pool

import data_utils as du

OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v3"
TEST_REPS = du.TEST_REPS
VAL_REPS = du.VAL_REPS

CFG = dict(win_ms=300, step_ms=150, min_purity=0.8)


def _worker(subject_id):
    t0 = time.time()
    X, y, r, s = du.load_and_segment_subject(subject_id, **CFG)
    print(f"  [S{subject_id:02d}] windows={len(y)} elapsed={time.time()-t0:.1f}s", flush=True)
    return X, y, r, s


def build(subjects=du.SUBJECTS, n_workers=8):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Config: {CFG}")
    print(f"Loading {len(subjects)} subjects with {n_workers} workers...")
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_worker, subjects)
    print(f"All subjects loaded in {time.time()-t0:.1f}s")

    Xs, ys, rs, ss = zip(*results)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    r_all = np.concatenate(rs, axis=0)
    s_all = np.concatenate(ss, axis=0)
    del Xs, ys, rs, ss, results
    print(f"Pooled: X={X_all.shape} ({X_all.nbytes/1e9:.1f} GB), classes={len(np.unique(y_all))}")

    test_mask = np.isin(r_all, TEST_REPS)
    val_mask = np.isin(r_all, VAL_REPS)
    train_mask = ~(test_mask | val_mask)

    splits = {"train": train_mask, "val": val_mask, "test": test_mask}

    mean = np.zeros((1, 1, X_all.shape[2]), dtype=np.float32)
    std = np.ones((1, 1, X_all.shape[2]), dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(OUT_DIR, "norm_std.npy"), std)

    for name, mask in splits.items():
        np.save(os.path.join(OUT_DIR, f"X_{name}.npy"), X_all[mask])
        np.save(os.path.join(OUT_DIR, f"y_{name}.npy"), y_all[mask])
        np.save(os.path.join(OUT_DIR, f"s_{name}.npy"), s_all[mask])
        print(f"  {name}: X={X_all[mask].shape} y_unique={len(np.unique(y_all[mask]))}")

    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    build()
