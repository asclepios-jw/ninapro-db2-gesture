"""
v9 1단계: 라벨-EMG 시차(반응시간) 보정값(delay_ms) 스윕.
v8에서 확정된 윈도우/순도 설정(300ms/150ms, purity>=0.8) 위에 delay_ms만 추가로 바꿔가며
8명 피험자 부분집합 + RandomForest(빠른 proxy)로 비교.
"""
import time
import json
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import data_utils as du
import features as feat

SUBJECTS = [1, 6, 11, 16, 21, 26, 31, 36]
DELAYS_MS = [0, 50, 100, 150, 200, 250]
BASE_CFG = dict(win_ms=300, step_ms=150, min_purity=0.8)

TEST_REPS = du.TEST_REPS
VAL_REPS = du.VAL_REPS


def _worker(args):
    subject_id, delay_ms = args
    X, y, r, s = du.load_and_segment_subject(subject_id, label_delay_ms=delay_ms, **BASE_CFG)
    return X, y, r, s


def run_delay(delay_ms, n_workers=8):
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_worker, [(sid, delay_ms) for sid in SUBJECTS])
    Xs, ys, rs, ss = zip(*results)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    r_all = np.concatenate(rs, axis=0)
    load_time = time.time() - t0

    test_mask = np.isin(r_all, TEST_REPS)
    val_mask = np.isin(r_all, VAL_REPS)
    train_mask = ~(test_mask | val_mask)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    t1 = time.time()
    F_train = feat.extract_features(X_train)
    F_val = feat.extract_features(X_val)
    feat_time = time.time() - t1

    t2 = time.time()
    clf = RandomForestClassifier(n_estimators=150, max_depth=20, n_jobs=-1, random_state=42,
                                  class_weight="balanced_subsample")
    clf.fit(F_train, y_train)
    train_time = time.time() - t2

    val_pred = clf.predict(F_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    return {
        "delay_ms": delay_ms,
        "n_train": int(len(y_train)), "n_val": int(len(y_val)),
        "val_accuracy": float(val_acc), "val_macro_f1": float(val_f1),
        "load_time_sec": load_time, "feat_time_sec": feat_time, "train_time_sec": train_time,
    }


def main():
    all_results = []
    for delay_ms in DELAYS_MS:
        print(f"Running delay_ms={delay_ms}")
        res = run_delay(delay_ms)
        print(f"  -> val_acc={res['val_accuracy']:.4f} val_macro_f1={res['val_macro_f1']:.4f} "
              f"(n_train={res['n_train']}, load={res['load_time_sec']:.1f}s)")
        all_results.append(res)

    all_results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    print("\n=== Ranking by val macro-F1 ===")
    for r in all_results:
        print(f"delay_ms={r['delay_ms']:>4d} -> acc={r['val_accuracy']:.4f} f1={r['val_macro_f1']:.4f}")

    with open("sweep_delay_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
