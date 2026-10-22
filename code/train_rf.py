"""RandomForest 베이스라인: 특징 추출 후 학습/평가."""
import os
import time
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import features as feat

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v2"
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v2"
os.makedirs(OUT_DIR, exist_ok=True)


def load_split(name):
    X = np.load(os.path.join(DATA_DIR, f"X_{name}.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{name}.npy"))
    return X, y


def main():
    print("Loading splits...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    print("Extracting features (train)...")
    t0 = time.time()
    F_train = feat.extract_features(X_train)
    print(f"  done in {time.time()-t0:.1f}s, shape={F_train.shape}")

    print("Extracting features (val/test)...")
    F_val = feat.extract_features(X_val)
    F_test = feat.extract_features(X_test)

    print("Training RandomForest...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, n_jobs=-1,
        random_state=42, class_weight="balanced_subsample",
    )
    clf.fit(F_train, y_train)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s")

    val_pred = clf.predict(F_val)
    test_pred = clf.predict(F_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"Val acc:  {val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}")

    report = classification_report(y_test, test_pred, zero_division=0)
    with open(os.path.join(OUT_DIR, "rf_report.txt"), "w") as f:
        f.write(report)

    result = {
        "model": "RandomForest",
        "n_estimators": 300,
        "max_depth": 20,
        "train_time_sec": train_time,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }
    with open(os.path.join(OUT_DIR, "rf_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    import joblib
    joblib.dump(clf, os.path.join(OUT_DIR, "rf_model.joblib"))
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
