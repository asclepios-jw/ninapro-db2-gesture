"""
v24: Multi-scale dilated Conv + BiLSTM (pooled 모델).

v23(Transformer+MaxPool) 실패 후 분리 검증:
  - Transformer 제거, MaxPool 제거 → BiLSTM 복귀
  - multi-scale conv(k=3/15/51) 단독 효과 측정
  - v8 대비 conv 블록만 교체, 나머지 동일

비교 기준: v10(pooled BiLSTM 50ep, 69.10%) — conv 블록만 다름.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from models_v24 import MultiScaleBiLSTMModel

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v24"
CKPT     = os.path.join(OUT_DIR, "best_model_v24.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}


class EMGDataset(Dataset):
    def __init__(self, X, y, s, mean, std):
        Xn = (X - mean) / std
        self.X = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def train_epoch(model, loader, optimizer):
    model.train()
    preds, labels = [], []
    for xb, yb, sb in loader:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        logits = model(xb, sb)
        loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        labels.extend(yb.cpu().numpy())
    return accuracy_score(labels, preds)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels, subs = [], [], []
    for xb, yb, sb in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        preds.extend(model(xb, sb).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
        subs.extend(sb.cpu().numpy())
    preds  = np.array(preds)
    labels = np.array(labels)
    subs   = np.array(subs)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    hard_idx = np.isin(subs + 1, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(labels[hard_idx],  preds[hard_idx])
    easy_acc = accuracy_score(labels[~hard_idx], preds[~hard_idx])
    return acc, macro_f1, hard_acc, easy_acc


def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()

    print("Loading data...")
    X_train   = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_r = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train   = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val     = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val_r   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val     = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test    = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_r  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test    = np.load(os.path.join(DATA_DIR, "s_test.npy"))
    mean      = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std       = np.load(os.path.join(DATA_DIR, "norm_std.npy"))

    le = LabelEncoder()
    le.fit(y_train_r)
    y_train = le.transform(y_train_r)
    y_val   = le.transform(y_val_r)
    y_test  = le.transform(y_test_r)
    n_classes  = len(le.classes_)
    n_channels = X_train.shape[2]
    n_subjects = int(s_train.max())

    print(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}")

    train_loader = DataLoader(EMGDataset(X_train, y_train, s_train - 1, mean, std),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EMGDataset(X_val,   y_val,   s_val   - 1, mean, std),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(EMGDataset(X_test,  y_test,  s_test  - 1, mean, std),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = MultiScaleBiLSTMModel(n_channels, n_classes, n_subjects).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"파라미터 수: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_val_f1  = -1.0
    patience_ctr = 0
    history      = []

    print(f"{'Ep':>3}  {'tr_acc':>7}  {'val_acc':>7}  {'val_f1':>7}  "
          f"{'hard':>7}  {'easy':>7}")

    for epoch in range(1, N_EPOCHS + 1):
        tr_acc = train_epoch(model, train_loader, optimizer)
        val_acc, val_f1, val_hard, val_easy = evaluate(model, val_loader)
        scheduler.step()

        flag = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CKPT)
            patience_ctr = 0
            flag = " ↑"
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

        print(f"{epoch:>3}  {tr_acc:>7.4f}  {val_acc:>7.4f}  {val_f1:>7.4f}  "
              f"{val_hard:>7.4f}  {val_easy:>7.4f}{flag}")

        history.append({"epoch": epoch, "train_acc": tr_acc,
                        "val_acc": val_acc, "val_macro_f1": val_f1,
                        "val_hard_acc": val_hard, "val_easy_acc": val_easy})

    print("\n[테스트 평가]")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    test_acc, test_f1, test_hard, test_easy = evaluate(model, test_loader)
    gap     = test_easy - test_hard
    elapsed = time.time() - t0

    print(f"Overall acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Macro-F1     : {test_f1:.4f}")
    print(f"Hard acc     : {test_hard:.4f}  ({test_hard*100:.2f}%)")
    print(f"Easy acc     : {test_easy:.4f}  ({test_easy*100:.2f}%)")
    print(f"Gap          : {gap:.4f}  ({gap*100:.2f}p)")
    print(f"\n비교 — v8  pooled 30ep             : 68.69%  hard 57.52%  easy 70.29%  gap 12.77p")
    print(f"비교 — v10 pooled 50ep (현재 최고)  : 69.10%  hard ~58%    easy ~71%    gap ~13p")
    print(f"비교 — v23 multi-scale+Transformer  : 67.73%  hard 58.21%  easy 69.09%  gap 10.88p")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v24_MultiScaleBiLSTM_pooled",
        "overall_accuracy": test_acc,
        "overall_macro_f1": test_f1,
        "hard_acc": test_hard,
        "easy_acc": test_easy,
        "gap": gap,
        "best_val_macro_f1": best_val_f1,
        "n_params": n_params,
        "elapsed_sec": elapsed,
        "hyperparams": {
            "n_epochs": N_EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "architecture": "MultiScale(k=3/15/51,d=1/2/4) + BiLSTM(128,2L,bidir) + attention + SubjEmb(32)",
        },
        "baselines": {
            "v8_pooled_30ep":  {"overall_accuracy": 0.6869, "hard_acc": 0.5752, "easy_acc": 0.7029, "gap": 0.1277},
            "v10_pooled_50ep": {"overall_accuracy": 0.6910},
            "v23_ms_transformer": {"overall_accuracy": 0.6773, "hard_acc": 0.5821, "easy_acc": 0.6909, "gap": 0.1088},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v24.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v24.json")


if __name__ == "__main__":
    main()
