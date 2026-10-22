"""CNN1D / CNN-LSTM 학습 및 평가 (raw EMG 윈도우 입력)."""
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from models import CNN1D, CNNLSTM

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v2"
OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v2"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMGDataset(Dataset):
    def __init__(self, X, y, mean, std):
        Xn = (X - mean) / std
        self.X = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))  # (N, C, T)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += len(yb)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
        truths.extend(yb.numpy())
    return np.array(truths), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn1d", "cnnlstm"], default="cnn1d")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std = np.load(os.path.join(DATA_DIR, "norm_std.npy"))

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape} n_classes={n_classes}")

    train_ds = EMGDataset(X_train, y_train_enc, mean, std)
    val_ds = EMGDataset(X_val, y_val_enc, mean, std)
    test_ds = EMGDataset(X_test, y_test_enc, mean, std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    n_channels = X_train.shape[2]
    model = (CNN1D(n_channels, n_classes) if args.model == "cnn1d" else CNNLSTM(n_channels, n_classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc, best_state, patience_ctr = 0.0, None, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[{args.model}] epoch {epoch:3d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, patience_ctr = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    y_true, y_pred = evaluate(model, test_loader)
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test acc: {test_acc:.4f}  Test macro-F1: {test_f1:.4f}")

    report = classification_report(y_true, y_pred, zero_division=0)
    with open(os.path.join(OUT_DIR, f"{args.model}_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(OUT_DIR, f"{args.model}_confusion_matrix.npy"), cm)

    result = {
        "model": args.model,
        "train_time_sec": train_time,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "n_epochs_run": len(history["train_loss"]),
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    }
    with open(os.path.join(OUT_DIR, f"{args.model}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    torch.save({
        "model_state": best_state,
        "model_config": {"n_channels": n_channels, "n_classes": n_classes},
        "norm_stats": {"mean": mean.tolist(), "std": std.tolist()},
        "label_classes": le.classes_.tolist(),
        "history": history,
        "test_accuracy": test_acc,
    }, os.path.join(OUT_DIR, f"{args.model}_best.pth"))
    print(f"Saved checkpoint to {OUT_DIR}/{args.model}_best.pth")


if __name__ == "__main__":
    main()
