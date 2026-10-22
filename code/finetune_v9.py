"""
v9: v8 베이스(pooled, 40명 동시 학습) 모델을 시작점으로, 각 피험자 본인의 train
데이터만으로 짧게 추가 학습(fine-tuning)한 뒤 그 피험자의 test 데이터로 평가.
40명 전체에 대해 반복하고 평균 성능을 pooled-only(v8)와 비교한다.

라벨-EMG 시차 보정(delay_ms)은 스윕 결과 delay_ms=0(보정 없음)이 가장 좋아서
v9에서는 적용하지 않음 -> v8 데이터셋(data_processed/v8)을 그대로 재사용.
"""
import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
BASE_CKPT = "/mnt/data/medical_project/ninapro_db2_project/results/v8/best_model_v8.pth"
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v9"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FT_EPOCHS = 15
FT_PATIENCE = 5
FT_LR = 1e-4
FT_BATCH = 128


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))  # (N, C, T)
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
    preds_all, labels_all = [], []

    with torch.set_grad_enabled(train):
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(yb.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")
    return total_loss / total_n, acc, f1


def main():
    print("Loading full v8 dataset once...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val_raw = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_raw = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test = np.load(os.path.join(DATA_DIR, "s_test.npy"))

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    n_classes = len(le.classes_)
    n_channels = X_train.shape[2]

    print("Loading v8 base checkpoint...")
    base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)

    subjects = sorted(np.unique(s_train).tolist())
    per_subject_results = []

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    t_start = time.time()
    for sid in subjects:
        t0 = time.time()
        tr_mask, va_mask, te_mask = s_train == sid, s_val == sid, s_test == sid

        train_ds = EMGDataset(X_train[tr_mask], y_train[tr_mask], s_train[tr_mask])
        val_ds = EMGDataset(X_val[va_mask], y_val[va_mask], s_val[va_mask])
        test_ds = EMGDataset(X_test[te_mask], y_test[te_mask], s_test[te_mask])

        train_loader = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
        model.load_state_dict(base_ckpt)
        optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)

        # 베이스(pooled) 모델 그대로 이 피험자 test에 평가 (fine-tuning 전 baseline)
        _, base_acc, base_f1 = run_epoch(model, test_loader, criterion)

        best_val_f1, best_state, patience_ctr = -1.0, None, 0
        for epoch in range(1, FT_EPOCHS + 1):
            run_epoch(model, train_loader, criterion, optimizer)
            _, val_acc, val_f1 = run_epoch(model, val_loader, criterion)
            if val_f1 > best_val_f1:
                best_val_f1, patience_ctr = val_f1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_ctr += 1
                if patience_ctr >= FT_PATIENCE:
                    break

        model.load_state_dict(best_state)
        _, ft_acc, ft_f1 = run_epoch(model, test_loader, criterion)

        elapsed = time.time() - t0
        print(f"[S{sid:02d}] n_train={tr_mask.sum()} base_acc={base_acc:.4f} base_f1={base_f1:.4f} "
              f"-> ft_acc={ft_acc:.4f} ft_f1={ft_f1:.4f} (epochs={epoch}, {elapsed:.1f}s)", flush=True)

        per_subject_results.append({
            "subject": int(sid), "n_train": int(tr_mask.sum()),
            "base_acc": base_acc, "base_f1": base_f1,
            "ft_acc": ft_acc, "ft_f1": ft_f1, "n_epochs": epoch,
        })

    total_time = time.time() - t_start

    base_accs = [r["base_acc"] for r in per_subject_results]
    base_f1s = [r["base_f1"] for r in per_subject_results]
    ft_accs = [r["ft_acc"] for r in per_subject_results]
    ft_f1s = [r["ft_f1"] for r in per_subject_results]

    summary = {
        "n_subjects": len(subjects),
        "total_time_sec": total_time,
        "pooled_v8_base_mean_acc": float(np.mean(base_accs)),
        "pooled_v8_base_mean_f1": float(np.mean(base_f1s)),
        "finetuned_mean_acc": float(np.mean(ft_accs)),
        "finetuned_mean_f1": float(np.mean(ft_f1s)),
        "finetuned_std_acc": float(np.std(ft_accs)),
        "finetuned_std_f1": float(np.std(ft_f1s)),
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(OUT_DIR, "finetune_per_subject.json"), "w") as f:
        json.dump(per_subject_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
