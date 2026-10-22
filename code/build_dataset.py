"""
40명 피험자(S1~S40) x E1+E2+E3 전체를 로딩/전처리/세그먼트하여
repetition 기준 train/val/test로 풀링(subject-dependent)한 뒤 디스크에 저장.

출력 위치: /mnt/data/medical_project/dataset/ninapro_db2_processed/
(원본 raw 데이터셋과 기존 노트북 산출물은 건드리지 않음)
"""
import os
import time
import numpy as np
from multiprocessing import Pool

import data_utils as du

OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v2"
TEST_REPS = (6,)
VAL_REPS = (5,)


def _worker(subject_id):
    t0 = time.time()
    X, y, r, s = du.load_and_segment_subject(subject_id)
    print(f"  [S{subject_id:02d}] windows={len(y)} elapsed={time.time()-t0:.1f}s", flush=True)
    return X, y, r, s


def build(subjects=du.SUBJECTS, n_workers=8):
    os.makedirs(OUT_DIR, exist_ok=True)
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

    splits = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }

    # 정규화는 data_utils.normalize_per_subject()에서 피험자별로 이미 적용됨
    # (train rep 통계 기준, 피험자 간 신호 진폭 차이를 제거). 여기서는 identity로 저장.
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
