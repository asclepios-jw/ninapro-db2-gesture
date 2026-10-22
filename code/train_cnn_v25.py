"""
v25: v10 아키텍처 + 전처리 팀 새 데이터 기준선.

변경사항:
  - 데이터: data_processed/v8 → 전처리전달 (팀 전처리)
  - 정규화: 새 데이터는 이미 per-subject 정규화 완료 → Dataset에서 정규화 미적용
  - 아키텍처/하이퍼파라미터: v10과 완전 동일

비교 기준: v10(69.10%, 우리 전처리) → 새 전처리에서 어느 수준인지 확인.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v25"
CKPT     = os.path.join(OUT_DIR, "best_model_v25.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}


# ── v8/v10과 동일한 아키텍처 ────────────────────────────────────────────────
class SubjectAwareCNNLSTM(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, n_subjects: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,        128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            128, 128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.attn     = nn.Linear(256, 1)
        self.subj_emb = nn.Embedding(n_subjects, 32)
        self.head     = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 + 32, n_classes),
        )

    def forward(self, x, s):
        h = self.conv(x)                          # (B, 128, T)
        h = h.transpose(1, 2)                     # (B, T, 128)
        h, _ = self.lstm(h)                       # (B, T, 256)
        a = torch.softmax(self.attn(h), dim=1)
        feat = (h * a).sum(dim=1)                 # (B, 256)
        emb  = self.subj_emb(s)                   # (B, 32)
        return self.head(torch.cat([feat, emb], dim=1))


# ── Dataset: 정규화 없음 (이미 per-subject 정규화 완료) ──────────────────────
class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
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
    subs   = np.array(subs)   # 0-indexed
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

    # 라벨 인코딩 (비연속 → 0~49)
    le = LabelEncoder()
    le.fit(y_train_r)
    y_train = le.transform(y_train_r)
    y_val   = le.transform(y_val_r)
    y_test  = le.transform(y_test_r)

    n_classes  = len(le.classes_)
    n_channels = X_train.shape[2]
    n_subjects = int(s_train.max())   # 40

    print(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}")
    print(f"데이터 정규화: 전처리 단계에서 per-subject 적용 완료 (추가 정규화 없음)\n")

    train_loader = DataLoader(EMGDataset(X_train, y_train, s_train - 1),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EMGDataset(X_val,   y_val,   s_val   - 1),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(EMGDataset(X_test,  y_test,  s_test  - 1),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SubjectAwareCNNLSTM(n_channels, n_classes, n_subjects).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"파라미터 수: {n_params:,}")

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
    print(f"\n비교 — v10 (우리 전처리, 동일 아키텍처) : 69.10%  hard ~58%  easy ~71%  gap ~13p")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v25_SubjectAwareCNNLSTM_new_preprocess",
        "overall_accuracy": test_acc,
        "overall_macro_f1": test_f1,
        "hard_acc": test_hard,
        "easy_acc": test_easy,
        "gap": gap,
        "best_val_macro_f1": best_val_f1,
        "n_params": n_params,
        "elapsed_sec": elapsed,
        "data_dir": DATA_DIR,
        "hyperparams": {
            "n_epochs": N_EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "normalization": "pre-applied per-subject by preprocessing team",
            "architecture": "Conv1d(24→64→128,k=3) + BiLSTM(128,2L,bidir) + attention + SubjEmb(32)",
        },
        "baselines": {
            "v10_our_preprocess": {"overall_accuracy": 0.6910, "gap": 0.13},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v25.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v25.json")


if __name__ == "__main__":
    main()
