"""
v20: Subject-aware DANN — subject embedding 유지 + hard/easy 이진 도메인 적응.

v18과 다른 점:
  - subject embedding(32차원)은 label head에 그대로 사용
  - 도메인 분류기는 40명 전체가 아니라 hard(0) / easy(1) 이진 분류
  - GRL은 256차원 backbone feature에만 작용 → backbone이 두 그룹에 공통인
    특징을 학습하도록 유도하면서, label head는 피험자별 정보까지 활용 가능
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

from models_v19 import SubjectAwareDANN

# ── 경로 ────────────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/data_processed/v8"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v20"
CKPT     = os.path.join(OUT_DIR, "best_model_v20.pth")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────────
N_EPOCHS      = 30
PATIENCE      = 6
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LAMBDA_DOMAIN = 0.1

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}   # 1-indexed, s 값 기준


def is_hard(s_val):
    return 0 if int(s_val) in HARD_SUBJECTS else 1  # hard=0, easy=1


# ── Dataset ─────────────────────────────────────────────────────────────────
class EMGDataset(Dataset):
    def __init__(self, X, y, s, domain, mean, std):
        Xn = (X - mean) / std
        self.X      = torch.from_numpy(Xn.transpose(0, 2, 1).astype(np.float32))
        self.y      = torch.from_numpy(y.astype(np.int64))
        self.s      = torch.from_numpy(s.astype(np.int64))
        self.domain = torch.from_numpy(domain.astype(np.int64))  # 0=hard, 1=easy

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx], self.domain[idx]


def get_alpha(epoch, n_epochs):
    p = epoch / n_epochs
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ── 학습 한 에포크 ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, alpha):
    model.train()
    total_cls = total_dom = total_n = 0
    all_preds, all_labels = [], []

    for xb, yb, sb, domain_b in loader:
        xb, yb, sb, domain_b = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE), domain_b.to(DEVICE)

        class_logits, domain_logits = model(xb, sb, alpha)

        loss_cls = F.cross_entropy(class_logits, yb, label_smoothing=0.1)
        loss_dom = F.cross_entropy(domain_logits, domain_b)
        loss = loss_cls + LAMBDA_DOMAIN * loss_dom

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cls += loss_cls.item() * len(yb)
        total_dom += loss_dom.item() * len(yb)
        total_n   += len(yb)
        all_preds.extend(class_logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    return total_cls / total_n, total_dom / total_n, accuracy_score(all_labels, all_preds)


# ── 검증 / 평가 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_subs = [], [], []

    for xb, yb, sb, _ in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        class_logits, _ = model(xb, sb, alpha=0.0)
        all_preds.extend(class_logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())
        all_subs.extend(sb.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_subs   = np.array(all_subs)

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    hard_idx = np.isin(all_subs, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(all_labels[hard_idx],  all_preds[hard_idx])
    easy_acc = accuracy_score(all_labels[~hard_idx], all_preds[~hard_idx])
    return acc, macro_f1, hard_acc, easy_acc


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()

    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val_raw   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val   = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_raw  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test  = np.load(os.path.join(DATA_DIR, "s_test.npy"))
    mean = np.load(os.path.join(DATA_DIR, "norm_mean.npy"))
    std  = np.load(os.path.join(DATA_DIR, "norm_std.npy"))

    n_channels = X_train.shape[2]
    n_subjects = len(np.unique(s_train))  # 40, s는 1-indexed

    # gesture 라벨: offset → 0-49
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    y_test  = le.transform(y_test_raw)
    n_classes = len(le.classes_)

    # 이진 도메인 라벨: hard=0, easy=1
    domain_train = np.array([is_hard(s) for s in s_train], dtype=np.int64)
    domain_val   = np.array([is_hard(s) for s in s_val],   dtype=np.int64)
    domain_test  = np.array([is_hard(s) for s in s_test],  dtype=np.int64)

    hard_ratio = (domain_train == 0).mean()
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}")
    print(f"Hard 비율: {hard_ratio:.2%}  Easy 비율: {1-hard_ratio:.2%}")

    train_ds = EMGDataset(X_train, y_train, s_train - 1, domain_train, mean, std)
    val_ds   = EMGDataset(X_val,   y_val,   s_val   - 1, domain_val,   mean, std)
    test_ds  = EMGDataset(X_test,  y_test,  s_test  - 1, domain_test,  mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SubjectAwareDANN(n_channels, n_classes, n_subjects).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    print("\n[학습 시작]")
    print(f"{'Ep':>3}  {'alpha':>5}  {'cls_loss':>9}  {'dom_loss':>9}  "
          f"{'tr_acc':>7}  {'val_acc':>7}  {'val_f1':>7}  {'hard_acc':>8}  {'easy_acc':>8}")

    for epoch in range(1, N_EPOCHS + 1):
        alpha = get_alpha(epoch - 1, N_EPOCHS)

        cls_loss, dom_loss, tr_acc = train_epoch(model, train_loader, optimizer, alpha)
        val_acc, val_f1, val_hard, val_easy = evaluate(model, val_loader)
        scheduler.step()

        print(f"{epoch:>3}  {alpha:>5.3f}  {cls_loss:>9.4f}  {dom_loss:>9.4f}  "
              f"{tr_acc:>7.4f}  {val_acc:>7.4f}  {val_f1:>7.4f}  "
              f"{val_hard:>8.4f}  {val_easy:>8.4f}")

        history.append({
            "epoch": epoch, "alpha": alpha,
            "cls_loss": cls_loss, "dom_loss": dom_loss,
            "train_acc": tr_acc, "val_acc": val_acc, "val_macro_f1": val_f1,
            "val_hard_acc": val_hard, "val_easy_acc": val_easy,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CKPT)
            patience_counter = 0
            print(f"    ↑ best val macro-F1 saved ({best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    print("\n[테스트 평가 — best checkpoint]")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    test_acc, test_f1, test_hard, test_easy = evaluate(model, test_loader)
    gap = test_easy - test_hard

    print(f"Overall acc  : {test_acc:.4f}")
    print(f"Macro-F1     : {test_f1:.4f}")
    print(f"Hard acc     : {test_hard:.4f}")
    print(f"Easy acc     : {test_easy:.4f}")
    print(f"Gap(easy-hard): {gap:.4f}")
    print(f"\n비교 — v8  : overall 0.6869 / F1 0.5692 / hard 0.5752 / easy 0.7029 / gap 0.1277")
    print(f"비교 — v9  : overall 0.6907 / hard ~0.58  / easy ~0.71  / gap ~0.13")
    print(f"비교 — v18 : overall 0.5910 / F1 0.3789 / hard 0.4747 / easy 0.6075 / gap +0.1328")
    print(f"비교 — v19 : overall 0.6227 / F1 0.4464 / hard 0.6833 / easy 0.6142 / gap -0.0690 (lambda=1.0)")

    elapsed = time.time() - t0
    result = {
        "model": "v20_SubjectAwareDANN",
        "overall_accuracy": test_acc,
        "overall_macro_f1": test_f1,
        "hard_acc": test_hard,
        "easy_acc": test_easy,
        "gap": gap,
        "best_val_macro_f1": best_val_f1,
        "elapsed_sec": elapsed,
        "hyperparams": {
            "n_epochs": N_EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "lambda_domain": LAMBDA_DOMAIN, "domain": "binary (hard=0, easy=1)",
        },
        "baselines": {
            "v8":  {"overall_accuracy": 0.6869, "overall_macro_f1": 0.5692, "hard_acc": 0.5752, "easy_acc": 0.7029, "gap": 0.1277},
            "v9":  {"overall_accuracy": 0.6907},
            "v18": {"overall_accuracy": 0.5910, "overall_macro_f1": 0.3789, "hard_acc": 0.4747, "easy_acc": 0.6075, "gap": 0.1328},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v20.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nElapsed: {elapsed/60:.1f}min  →  결과 저장: {OUT_DIR}/result_v20.json")


if __name__ == "__main__":
    main()
