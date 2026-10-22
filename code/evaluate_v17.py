"""
v17: v16의 하드 라우팅(1단계 top-1 클러스터만 믿고 2단계로 넘김) 대신
소프트 라우팅(1단계의 클러스터별 확률을 그대로 살려서 2단계 전문가들의
예측을 가중 합산)을 시도. 재학습 없이 v15(1단계)/v16(2단계) 체크포인트를
그대로 재사용 — 추론만 다르게 한다.

방식: 모든 test 윈도우에 대해
  1) 1단계 모델로 10개 클러스터에 대한 softmax 확률 p1 (N x 10)을 구한다.
  2) 클러스터 c마다(트리비얼 클러스터 포함) 그 클러스터의 2단계 전문가로
     "전역 50클래스 중 자기 클러스터에 속한 클래스들"에 대한 확률을 구하고,
     p1[:, c]를 곱해서 전역 확률 분포(N x 50)에 더한다 (mixture of experts).
  3) 모든 클러스터의 기여를 합산한 뒤 argmax로 최종 예측.
하드 라우팅과 달리, 1단계가 애매하게 두 클러스터 사이에서 흔들렸다면 그
두 클러스터의 2단계 의견이 모두 일정 비중으로 반영된다.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
V14_RESULTS = "/mnt/data/medical_project/ninapro_db2_project/results/v14/class_grouping_results.json"
STAGE1_CKPT = "/mnt/data/medical_project/ninapro_db2_project/results/v15/best_model_v15.pth"
V16_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v16"
V17_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v17"
os.makedirs(V17_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
N_GLOBAL_CLASSES = 50


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
def predict_proba(model, X, s, mean, std, batch_size=1024):
    ds = EMGSubjectDataset(X, s, mean, std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    for xb, sb in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        logits = model(xb, sb)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


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
    y_test_enc = le.transform(y_test_raw)

    cluster_assignment = np.array(json.load(open(V14_RESULTS))["k10"]["cluster_assignment"])
    n_clusters = len(np.unique(cluster_assignment))
    meta = json.load(open(os.path.join(V16_DIR, "stage2_meta.json")))
    cluster_info = {c["cluster"]: c for c in meta["clusters"]}

    print("Stage 1: computing cluster probabilities for every test window...")
    stage1 = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_clusters, n_subjects=40).to(DEVICE)
    stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=True))
    p1 = predict_proba(stage1, X_test, s_test, mean, std)  # (N, 10)
    del stage1
    torch.cuda.empty_cache()

    n = len(y_test_enc)
    global_proba = np.zeros((n, N_GLOBAL_CLASSES), dtype=np.float64)

    print("Stage 2: running every sub-classifier on the FULL test set (mixture of experts)...")
    for c in range(n_clusters):
        info = cluster_info[c]
        weight = p1[:, c]  # (N,)
        if (weight < 1e-6).all():
            continue

        if info["trivial"]:
            global_label = info["global_labels"][0]
            global_proba[:, global_label] += weight
            print(f"[cluster {c}] trivial, contributed to global label {global_label}")
            continue

        n_classes = info["n_classes"]
        global_labels = np.array(info["global_labels"])
        model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
        ckpt_path = os.path.join(V16_DIR, f"stage2_cluster{c}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        p2_local = predict_proba(model, X_test, s_test, mean, std)  # (N, n_classes)
        global_proba[:, global_labels] += weight[:, None] * p2_local
        print(f"[cluster {c}] n_classes={n_classes} mixed in (avg weight={weight.mean():.4f})")
        del model
        torch.cuda.empty_cache()

    final_pred = global_proba.argmax(axis=1)

    overall_acc = accuracy_score(y_test_enc, final_pred)
    overall_f1 = f1_score(y_test_enc, final_pred, average="macro")
    hard_idx = np.isin(s_test, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(y_test_enc[hard_idx], final_pred[hard_idx])
    easy_acc = accuracy_score(y_test_enc[~hard_idx], final_pred[~hard_idx])
    gap = easy_acc - hard_acc

    print(f"\n=== v17 (소프트 라우팅) 최종 결과 ===")
    print(f"Overall acc: {overall_acc:.4f}  Macro-F1: {overall_f1:.4f}")
    print(f"Hard subjects acc: {hard_acc:.4f}  Easy subjects acc: {easy_acc:.4f}  Gap: {gap:.4f}")
    print(f"(비교: v16 하드라우팅 -> overall 0.6802, hard 0.5604, easy 0.6972, gap 0.1368)")
    print(f"(비교: v8 단일 50클래스 -> overall 0.6869, hard 0.5752, easy 0.7029, gap 0.1277)")

    result = {
        "model": "v17_soft_routing_mixture",
        "overall_accuracy": overall_acc, "overall_macro_f1": overall_f1,
        "hard_acc": hard_acc, "easy_acc": easy_acc, "gap": gap,
        "v16_hard_routing": {"overall_accuracy": 0.6802, "hard_acc": 0.5604, "easy_acc": 0.6972, "gap": 0.1368},
        "v8_baseline": {"overall_accuracy": 0.686940122583687, "overall_macro_f1": 0.5692153658106086,
                         "hard_acc": 0.5752002373183032, "easy_acc": 0.7028525563171291, "gap": 0.1276523189988259},
    }
    with open(os.path.join(V17_DIR, "result_v17.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {V17_DIR}")


if __name__ == "__main__":
    main()
