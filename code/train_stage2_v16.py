"""
v16: 2단계 계층적 분류기의 2단계 — 각 상위클래스(cluster) 내부에서 원래
세부 클래스(fine label)를 다시 분류하는 서브 분류기 10개를 학습.
1단계(상위클래스 10개 예측)는 v15 체크포인트를 그대로 재사용.

cluster 6은 클래스가 1개뿐이라 학습이 필요 없음(항상 그 클래스를 예측).
나머지 9개 클러스터는 그 클러스터에 속하는 윈도우만 모아서, 같은 아키텍처/
레시피로 "그 클러스터 내부의 fine label"을 맞히는 분류기를 학습한다.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
V14_RESULTS = "/mnt/data/medical_project/ninapro_db2_project/results/v14/class_grouping_results.json"
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v16"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMGSubjectDataset(Dataset):
    def __init__(self, X, y, s, mean, std):
        Xn = (X - mean) / std
        self.X = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / total_n, acc, all_labels, all_preds


def train_one_cluster(cluster_id, global_labels, X_train, y_train_enc, s_train, mean, std,
                       X_val, y_val_enc, s_val, n_channels):
    train_mask = np.isin(y_train_enc, global_labels)
    val_mask = np.isin(y_val_enc, global_labels)

    local_le = LabelEncoder()
    local_le.fit(global_labels)  # 순서 고정: global_labels 그대로 정렬된 순서

    y_train_local = local_le.transform(y_train_enc[train_mask])
    y_val_local = local_le.transform(y_val_enc[val_mask])
    n_classes = len(local_le.classes_)

    train_ds = EMGSubjectDataset(X_train[train_mask], y_train_local, s_train[train_mask], mean, std)
    val_ds = EMGSubjectDataset(X_val[val_mask], y_val_local, s_val[val_mask], mean, std)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs, patience = 30, 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, criterion, optimizer)
        _, val_acc, val_labels, val_preds = run_epoch(model, val_loader, criterion)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step()
        if val_f1 > best_val_f1:
            best_val_f1, patience_ctr = val_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    elapsed = time.time() - t0
    torch.save(best_state, os.path.join(OUT_DIR, f"stage2_cluster{cluster_id}.pth"))
    print(f"[cluster {cluster_id}] n_classes={n_classes} n_train={train_mask.sum()} "
          f"best_val_f1={best_val_f1:.4f} epochs={epoch} time={elapsed:.1f}s", flush=True)
    return {"cluster": cluster_id, "n_classes": n_classes, "global_labels": local_le.classes_.tolist(),
            "n_train": int(train_mask.sum()), "best_val_f1": best_val_f1, "n_epochs": epoch,
            "train_time_sec": elapsed}


def main():
    cluster_assignment = np.array(json.load(open(V14_RESULTS))["k10"]["cluster_assignment"])
    n_clusters = len(np.unique(cluster_assignment))

    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val_raw = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(DATA_DIR, "norm_std.npy"))
    n_channels = X_train.shape[2]

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train_enc = le.transform(y_train_raw)
    y_val_enc = le.transform(y_val_raw)

    meta = {"label_encoder_classes": le.classes_.tolist(), "clusters": []}
    for c in range(n_clusters):
        global_labels = np.where(cluster_assignment == c)[0]
        if len(global_labels) == 1:
            print(f"[cluster {c}] n_classes=1 -> trivial (no training needed, "
                  f"always predict global label {global_labels[0]})", flush=True)
            meta["clusters"].append({"cluster": c, "n_classes": 1,
                                      "global_labels": global_labels.tolist(), "trivial": True})
            continue
        info = train_one_cluster(c, global_labels, X_train, y_train_enc, s_train, mean, std,
                                  X_val, y_val_enc, s_val, n_channels)
        info["trivial"] = False
        meta["clusters"].append(info)

    with open(os.path.join(OUT_DIR, "stage2_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved stage2 metadata to {OUT_DIR}")


if __name__ == "__main__":
    main()
