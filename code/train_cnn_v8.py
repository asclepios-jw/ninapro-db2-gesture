"""
v8 최종: 300ms/150ms 윈도우 + 라벨 순도>=0.8 필터 + rectification/RMS envelope 채널(24ch)
+ v6의 검증된 레시피(SubjectAware+Attention, CE+label_smoothing, val macro-F1 체크포인트).
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
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v8"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMGSubjectDataset(Dataset):
    def __init__(self, X, y, s, mean, std):
        Xn = (X - mean) / std
        self.X = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))  # (N, C, T)
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
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True), sb.to(DEVICE, non_blocking=True)
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

    avg_loss = total_loss / total_n
    acc = total_correct / total_n

    if not train:
        return avg_loss, acc, all_labels, all_preds
    return avg_loss, acc


def main():
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))

    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val = np.load(os.path.join(DATA_DIR, "s_val.npy"))

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test = np.load(os.path.join(DATA_DIR, "s_test.npy"))

    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(DATA_DIR, "norm_std.npy"))

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)
    n_channels = X_train.shape[2]
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape} n_classes={n_classes} n_channels={n_channels}")

    train_ds = EMGSubjectDataset(X_train, y_train_enc, s_train, mean, std)
    val_ds = EMGSubjectDataset(X_val, y_val_enc, s_val, mean, std)
    test_ds = EMGSubjectDataset(X_test, y_test_enc, s_test, mean, std)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    print("Initializing Model V8 (SubjectAware + Attention, 24ch raw+envelope)...")
    model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs, patience = 30, 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

    print(f"Start training on {DEVICE}...")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_labels, val_preds = run_epoch(model, val_loader, criterion)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} MacroF1: {val_macro_f1:.4f} | "
              f"Time: {time.time()-ep_t0:.1f}s")

        if val_macro_f1 > best_val_f1:
            best_val_f1, patience_ctr = val_macro_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(OUT_DIR, "best_model_v8.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    print("Evaluating Best Model on Test Set...")
    test_loss, test_acc, test_labels, test_preds = run_epoch(model, test_loader, criterion)
    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    weighted_f1 = f1_score(test_labels, test_preds, average="weighted")

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    print(f"Test Weighted-F1: {weighted_f1:.4f}")

    with open(os.path.join(OUT_DIR, "test_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Test Weighted-F1: {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, test_preds, digits=4))

    cm = confusion_matrix(test_labels, test_preds)
    np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)

    result = {
        "model": "SubjectAwareCNNLSTMAttention_v8",
        "config": "win=300ms/step=150ms/min_purity=0.8/add_envelope=True",
        "train_time_sec": train_time,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "n_epochs_run": len(history["train_loss"]),
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    }
    with open(os.path.join(OUT_DIR, "result_v8.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
