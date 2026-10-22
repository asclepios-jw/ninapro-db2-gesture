"""
v22: 40명 개별 모델 — 에포크 50으로 증가 (v21: 30epoch → v22: 50epoch).

v21에서 다수 피험자가 patience 소진 없이 30epoch를 완주함.
더 긴 학습으로 추가 개선 여지 확인.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
OUT_DIR   = "/mnt/data/medical_project/ninapro_db2_project/results/v22"
CKPT_PATH = os.path.join(OUT_DIR, "_tmp_ckpt.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}


class SingleSubjectModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(256, 1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        h = self.conv(x)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        a = torch.softmax(self.attn(h), dim=1)
        return self.head((h * a).sum(dim=1))


class SimpleDataset(Dataset):
    def __init__(self, X, y, mean, std):
        Xn = (X - mean) / std
        self.X = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_one_subject(X_tr, y_tr, X_va, y_va, X_te, y_te, n_channels):
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-6

    le = LabelEncoder()
    le.fit(y_tr)
    y_tr_enc = le.transform(y_tr)
    y_va_enc = le.transform(y_va)
    y_te_enc = le.transform(y_te)
    n_classes = len(le.classes_)

    tr_loader = DataLoader(SimpleDataset(X_tr, y_tr_enc, mean, std),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(SimpleDataset(X_va, y_va_enc, mean, std),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    te_loader = DataLoader(SimpleDataset(X_te, y_te_enc, mean, std),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    model = SingleSubjectModel(n_channels, n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_val_f1  = -1.0
    best_epoch   = 0
    patience_ctr = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                labels.extend(yb.numpy())
        val_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1, best_val_f1, best_epoch


def main():
    print(f"Device: {DEVICE}")
    print(f"N_EPOCHS={N_EPOCHS}  PATIENCE={PATIENCE}")
    t0 = time.time()

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val   = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test  = np.load(os.path.join(DATA_DIR, "s_test.npy"))

    n_channels = X_train.shape[2]
    subjects   = sorted(np.unique(s_train).tolist())

    print(f"피험자 {len(subjects)}명 개별 학습 (n_channels={n_channels})\n")
    print(f"{'Sub':>4}  {'tr_win':>7}  {'te_acc':>7}  {'te_f1':>7}  {'best_ep':>7}  {'sec':>5}  {'type':>6}")

    per_subject = []

    for s in subjects:
        s_t0    = time.time()
        mask_tr = s_train == s
        mask_va = s_val   == s
        mask_te = s_test  == s

        acc, f1, best_vf1, best_ep = train_one_subject(
            X_train[mask_tr], y_train[mask_tr],
            X_val[mask_va],   y_val[mask_va],
            X_test[mask_te],  y_test[mask_te],
            n_channels,
        )
        elapsed_s = time.time() - s_t0
        label = "HARD" if s in HARD_SUBJECTS else "easy"
        print(f"{s:>4}  {mask_tr.sum():>7,}  {acc:>7.4f}  {f1:>7.4f}  "
              f"{best_ep:>7}  {elapsed_s:>5.0f}  {label:>6}")

        per_subject.append({
            "subject": int(s), "is_hard": s in HARD_SUBJECTS,
            "train_windows": int(mask_tr.sum()),
            "test_acc": acc, "test_macro_f1": f1,
            "best_val_macro_f1": best_vf1, "best_epoch": best_ep,
        })
        torch.cuda.empty_cache()

    accs      = [r["test_acc"] for r in per_subject]
    hard_accs = [r["test_acc"] for r in per_subject if r["is_hard"]]
    easy_accs = [r["test_acc"] for r in per_subject if not r["is_hard"]]

    avg_acc  = np.mean(accs)
    hard_avg = np.mean(hard_accs)
    easy_avg = np.mean(easy_accs)
    gap      = easy_avg - hard_avg
    elapsed  = time.time() - t0

    print(f"\n{'='*55}")
    print(f"[최종 결과]")
    print(f"전체 평균 acc (40명)  : {avg_acc:.4f}  ({avg_acc*100:.2f}%)")
    print(f"Hard 5명 평균 acc     : {hard_avg:.4f}  ({hard_avg*100:.2f}%)")
    print(f"Easy 35명 평균 acc    : {easy_avg:.4f}  ({easy_avg*100:.2f}%)")
    print(f"Gap (easy - hard)     : {gap:.4f}  ({gap*100:.2f}p)")
    print(f"\n비교 — v8  (pooled 30ep)  : 68.69%  hard 57.52%  easy 70.29%  gap 12.77p")
    print(f"비교 — v10 (pooled 50ep)  : 69.10%")
    print(f"비교 — v21 (indiv  30ep)  : 69.00%  hard 57.59%  easy 70.63%  gap 13.04p")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v22_individual_40subjects_50ep",
        "avg_test_acc": avg_acc,
        "hard_avg_acc": hard_avg,
        "easy_avg_acc": easy_avg,
        "gap": gap,
        "elapsed_sec": elapsed,
        "hyperparams": {"n_epochs": N_EPOCHS, "patience": PATIENCE,
                        "batch_size": BATCH_SIZE, "lr": LR},
        "per_subject": per_subject,
        "baselines": {
            "v8_pooled_30ep":  {"overall_accuracy": 0.6869, "hard_acc": 0.5752, "easy_acc": 0.7029, "gap": 0.1277},
            "v10_pooled_50ep": {"overall_accuracy": 0.6910},
            "v21_indiv_30ep":  {"overall_accuracy": 0.6900, "hard_acc": 0.5759, "easy_acc": 0.7063, "gap": 0.1304},
        }
    }
    out = os.path.join(OUT_DIR, "result_v22.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
