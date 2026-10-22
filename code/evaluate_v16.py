"""
v16: 2단계 계층적 분류기 전체 평가.
1단계(v15, 10개 상위클래스 예측) -> 그 예측에 해당하는 2단계 서브 분류기로 라우팅
-> 서브 분류기가 원래 50개 fine label 중 자기 클러스터 내부의 label을 예측
-> 전역 fine label로 역매핑 -> v8(단일 50클래스 모델)과 정확도 비교.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
V14_RESULTS = "/mnt/data/medical_project/ninapro_db2_project/results/v14/class_grouping_results.json"
STAGE1_CKPT = "/mnt/data/medical_project/ninapro_db2_project/results/v15/best_model_v15.pth"
V16_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v16"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}


class EMGSubjectDataset(Dataset):
    def __init__(self, X, s, mean, std):
        Xn = (X - mean) / std
        self.X = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))
        self.s = torch.from_numpy(s.astype(np.int64))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.X[idx], self.s[idx]


@torch.no_grad()
def predict(model, X, s, mean, std, batch_size=1024):
    ds = EMGSubjectDataset(X, s, mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    for xb, sb in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        logits = model(xb, sb)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_raw = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test = np.load(os.path.join(DATA_DIR, "s_test.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(DATA_DIR, "norm_std.npy"))
    n_channels = X_test.shape[2]

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_test_enc = le.transform(y_test_raw)  # 전역 fine label (0-49 인코딩)

    cluster_assignment = np.array(json.load(open(V14_RESULTS))["k10"]["cluster_assignment"])
    n_clusters = len(np.unique(cluster_assignment))
    meta = json.load(open(os.path.join(V16_DIR, "stage2_meta.json")))
    cluster_info = {c["cluster"]: c for c in meta["clusters"]}

    print("Stage 1: predicting superclass for every test window...")
    stage1 = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_clusters, n_subjects=40).to(DEVICE)
    stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=True))
    coarse_pred = predict(stage1, X_test, s_test, mean, std)
    del stage1
    torch.cuda.empty_cache()

    final_pred = np.full(len(y_test_enc), -1, dtype=np.int64)

    print("Stage 2: routing each window to its predicted cluster's sub-classifier...")
    for c in range(n_clusters):
        mask = coarse_pred == c
        if mask.sum() == 0:
            continue
        info = cluster_info[c]
        if info["trivial"]:
            final_pred[mask] = info["global_labels"][0]
            print(f"[cluster {c}] trivial -> {mask.sum()} windows assigned label {info['global_labels'][0]}")
            continue

        n_classes = info["n_classes"]
        global_labels = np.array(info["global_labels"])
        model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
        ckpt_path = os.path.join(V16_DIR, f"stage2_cluster{c}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        local_pred = predict(model, X_test[mask], s_test[mask], mean, std)
        final_pred[mask] = global_labels[local_pred]
        print(f"[cluster {c}] {mask.sum()} windows routed, n_classes={n_classes}")
        del model
        torch.cuda.empty_cache()

    assert (final_pred >= 0).all(), "일부 윈도우가 어떤 클러스터로도 라우팅되지 않음"

    overall_acc = accuracy_score(y_test_enc, final_pred)
    overall_f1 = f1_score(y_test_enc, final_pred, average="macro")

    hard_idx = np.isin(s_test, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(y_test_enc[hard_idx], final_pred[hard_idx])
    easy_acc = accuracy_score(y_test_enc[~hard_idx], final_pred[~hard_idx])
    gap = easy_acc - hard_acc

    print(f"\n=== v16 (2단계 계층적 분류기) 최종 결과 ===")
    print(f"Overall acc: {overall_acc:.4f}  Macro-F1: {overall_f1:.4f}")
    print(f"Hard subjects acc: {hard_acc:.4f}  Easy subjects acc: {easy_acc:.4f}  Gap: {gap:.4f}")
    print(f"(비교: v8 단일 50클래스 모델 -> overall 0.6869, macro-F1 0.5692, "
          f"hard 0.5752, easy 0.7029, gap 0.1277)")

    cm = confusion_matrix(y_test_enc, final_pred, labels=list(range(50)))
    np.save(os.path.join(V16_DIR, "confusion_matrix.npy"), cm)

    result = {
        "model": "v16_two_stage_hierarchical",
        "overall_accuracy": overall_acc, "overall_macro_f1": overall_f1,
        "hard_acc": hard_acc, "easy_acc": easy_acc, "gap": gap,
        "v8_baseline": {"overall_accuracy": 0.686940122583687, "overall_macro_f1": 0.5692153658106086,
                         "hard_acc": 0.5752002373183032, "easy_acc": 0.7028525563171291, "gap": 0.1276523189988259},
    }
    with open(os.path.join(V16_DIR, "result_v16.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {V16_DIR}")


if __name__ == "__main__":
    main()
