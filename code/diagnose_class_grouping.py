"""
v14: separability가 낮은 피험자에게 "클래스를 단순화(그룹화)"하면 도움이 되는지
진단. 모델을 새로 학습하지 않고 v8 베이스 모델의 test 예측을 한 번만 뽑아서,
1) 어떤 클래스들이 서로 자주 혼동되는지(confusion 기반 거리)로 계층적 클러스터링
2) 50개 클래스를 더 적은 수의 "상위 클래스"로 묶었을 때 정확도가 얼마나 오르는지
3) 그 상승분이 어려운 피험자(S28,S31 등)에게 더 크게 나타나는지(=문제가 "같은
   계열 안에서의 세부 혼동"이었는지) 아니면 고르게 나타나는지(=더 근본적인 문제)
를 확인한다.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
CKPT = "/mnt/data/medical_project/ninapro_db2_project/results/v8/best_model_v8.pth"
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v14"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
N_SUPERCLASSES_LIST = [10, 15, 20, 25]


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]


def main():
    print("Loading test data + v8 checkpoint...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_raw = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test = np.load(os.path.join(DATA_DIR, "s_test.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_test = le.transform(y_test_raw)
    n_classes = len(le.classes_)
    n_channels = X_test.shape[2]

    model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    ds = EMGDataset(X_test, y_test, s_test)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    preds, labels, subs = [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            xb, sb = xb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
            subs.extend(sb.cpu().numpy())
    preds, labels, subs = np.array(preds), np.array(labels), np.array(subs)

    fine_acc = accuracy_score(labels, preds)
    print(f"Fine-grained (50-class) overall accuracy: {fine_acc:.4f} (sanity check vs v8: 0.6869)")

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    sim = (cm_norm + cm_norm.T) / 2
    np.fill_diagonal(sim, 1.0)
    dist = 1 - sim
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist, checks=False), method="average")

    def subject_acc(y_true, y_pred, sid_arr, sid):
        mask = sid_arr == sid
        return accuracy_score(y_true[mask], y_pred[mask])

    hard_idx = np.isin(subs, list(HARD_SUBJECTS))
    results = {"fine_grained": {
        "overall_acc": float(fine_acc),
        "hard_acc": float(accuracy_score(labels[hard_idx], preds[hard_idx])),
        "easy_acc": float(accuracy_score(labels[~hard_idx], preds[~hard_idx])),
    }}

    for k in N_SUPERCLASSES_LIST:
        cluster_ids = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
        y_super = cluster_ids[labels]
        p_super = cluster_ids[preds]
        overall = accuracy_score(y_super, p_super)
        hard_acc = accuracy_score(y_super[hard_idx], p_super[hard_idx])
        easy_acc = accuracy_score(y_super[~hard_idx], p_super[~hard_idx])
        gap = easy_acc - hard_acc
        print(f"k={k:2d} superclasses | overall={overall:.4f} hard={hard_acc:.4f} "
              f"easy={easy_acc:.4f} gap={gap:.4f}")
        results[f"k{k}"] = {"overall_acc": float(overall), "hard_acc": float(hard_acc),
                             "easy_acc": float(easy_acc), "gap": float(gap),
                             "cluster_assignment": cluster_ids.tolist()}

    fine_gap = results["fine_grained"]["easy_acc"] - results["fine_grained"]["hard_acc"]
    print(f"\nFine-grained gap (easy - hard): {fine_gap:.4f}")
    for k in N_SUPERCLASSES_LIST:
        print(f"k={k} gap: {results[f'k{k}']['gap']:.4f} "
              f"(fine-grained 대비 {100*(1 - results[f'k{k}']['gap']/fine_gap):.1f}% 감소)")

    with open(os.path.join(OUT_DIR, "class_grouping_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
