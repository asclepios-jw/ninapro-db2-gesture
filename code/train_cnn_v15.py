"""
v15: v14에서 사후(post-hoc) 추정한 k=10 클래스 그룹화 효과가 실제 학습에서도
재현되는지 검증. v14의 cluster_assignment(50->10)로 v8 데이터의 라벨을
다시 매핑해서, 같은 아키텍처(SubjectAwareCNNLSTMAttention)와 학습 레시피로
"처음부터 10개 상위 클래스만 구분하는" 모델을 새로 학습.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
V14_RESULTS = "/mnt/data/medical_project/ninapro_db2_project/results/v14/class_grouping_results.json"
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v15"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}


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
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_labels, all_subs = [], [], []

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
            preds = logits.argmax(1)
            total_correct += (preds == yb).sum().item()
            total_n += len(yb)
            if not train:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                all_subs.extend(sb.cpu().numpy())

    if not train:
        return total_loss / total_n, total_correct / total_n, all_labels, all_preds, all_subs
    return total_loss / total_n, total_correct / total_n


def main():
    cluster_assignment = np.array(json.load(open(V14_RESULTS))["k10"]["cluster_assignment"])
    n_super = len(np.unique(cluster_assignment))
    print(f"Cluster assignment (50->{n_super}): {cluster_assignment.tolist()}")

    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val_raw = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_raw = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test = np.load(os.path.join(DATA_DIR, "s_test.npy"))
    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(DATA_DIR, "norm_std.npy"))

    # 주의: y_*_raw는 0-49 연속이 아니라 원본 라벨 오프셋(0-17,35-57,81-89)을 그대로
    # 담고 있다. v14의 cluster_assignment는 LabelEncoder로 인코딩한 0-49 순서를
    # 기준으로 계산됐으므로, 먼저 같은 방식으로 인코딩한 뒤에 매핑해야 한다.
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train_enc = le.transform(y_train_raw)
    y_val_enc = le.transform(y_val_raw)
    y_test_enc = le.transform(y_test_raw)

    y_train = cluster_assignment[y_train_enc]
    y_val = cluster_assignment[y_val_enc]
    y_test = cluster_assignment[y_test_enc]
    n_classes = n_super
    n_channels = X_train.shape[2]
    print(f"train={X_train.shape} n_classes={n_classes} n_channels={n_channels}")

    train_ds = EMGSubjectDataset(X_train, y_train, s_train, mean, std)
    val_ds = EMGSubjectDataset(X_val, y_val, s_val, mean, std)
    test_ds = EMGSubjectDataset(X_test, y_test, s_test, mean, std)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs, patience = 30, 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

    print(f"Start training on {DEVICE} (n_classes={n_classes})...")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_labels, val_preds, _ = run_epoch(model, val_loader, criterion)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step()

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} MacroF1 {val_macro_f1:.4f} | "
              f"Time {time.time()-ep_t0:.1f}s")

        if val_macro_f1 > best_val_f1:
            best_val_f1, patience_ctr = val_macro_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(OUT_DIR, "best_model_v15.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    test_loss, test_acc, test_labels, test_preds, test_subs = run_epoch(model, test_loader, criterion)
    test_labels, test_preds, test_subs = np.array(test_labels), np.array(test_preds), np.array(test_subs)
    test_f1 = f1_score(test_labels, test_preds, average="macro")

    hard_idx = np.isin(test_subs, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(test_labels[hard_idx], test_preds[hard_idx])
    easy_acc = accuracy_score(test_labels[~hard_idx], test_preds[~hard_idx])
    gap = easy_acc - hard_acc

    print(f"Test Acc: {test_acc:.4f}  Test Macro-F1: {test_f1:.4f}")
    print(f"Hard subjects acc: {hard_acc:.4f}  Easy subjects acc: {easy_acc:.4f}  Gap: {gap:.4f}")

    report = classification_report(test_labels, test_preds, zero_division=0)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write(report)
    cm = confusion_matrix(test_labels, test_preds)
    np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)

    result = {
        "model": "SubjectAwareCNNLSTMAttention_v15_k10superclass",
        "n_classes": n_classes,
        "train_time_sec": train_time,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "hard_acc": hard_acc, "easy_acc": easy_acc, "gap": gap,
        "n_epochs_run": len(history["train_loss"]),
        "posthoc_v14_k10": {"overall": 0.8591581847260357, "hard": 0.7874517947196678,
                             "easy": 0.8693696072322494, "gap": 0.08191781251258168},
    }
    with open(os.path.join(OUT_DIR, "result_v15.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUT_DIR, "history_v15.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
