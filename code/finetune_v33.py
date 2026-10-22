"""
v33: v32 체크포인트 기반 per-subject fine-tuning.

목적: v32(68.70%)에 v31 방식 fine-tuning 적용으로 69% 돌파 시도.
      v31(v30 base)에서 +0.13p였으나 v32는 hard acc가 v30보다 높아
      fine-tuning 과적합 위험이 일부 완화될 수 있음.

비교 기준:
  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)
  v32  68.70%  v28 전처리 + v8 arch (pooled)
  v31  68.38%  팀 10Hz + v8 arch (fine-tuned, v30 base)
  v33     ?    v28 전처리 + v8 arch (fine-tuned, v32 base)

데이터: 전처리전달_v28/ (이미 정규화 완료)
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR  = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
BASE_CKPT = "/mnt/data/medical_project/ninapro_db2_project/results/v32/best_model_v32.pth"
OUT_DIR   = "/mnt/data/medical_project/ninapro_db2_project/results/v33"
os.makedirs(OUT_DIR, exist_ok=True)

FT_EPOCHS   = 15
FT_PATIENCE = 5
FT_LR       = 1e-4
FT_BATCH    = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}   # 1-indexed

FREEZE_MODULES = ["cnn"]


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))   # 1-indexed

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def apply_freeze(model, freeze_modules):
    for name, module in model.named_children():
        if name in freeze_modules:
            for p in module.parameters():
                p.requires_grad = False
        else:
            for p in module.parameters():
                p.requires_grad = True


def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    preds, labels = [], []
    total_loss = 0.0
    ctx = torch.no_grad() if optimizer is None else torch.enable_grad()
    with ctx:
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            loss   = criterion(logits, yb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            preds.extend(logits.detach().argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(labels), acc, f1


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print(f"BASE_CKPT: {BASE_CKPT}\n")

    print("데이터 로드 중...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val   = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test  = np.load(os.path.join(DATA_DIR, "s_test.npy"))

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)

    n_classes  = len(le.classes_)
    n_channels = X_train.shape[2]
    n_subjects = int(s_train.max())   # 40

    print(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    subject_ids = sorted(np.unique(s_train))

    base_accs, ft_accs = [], []
    base_f1s,  ft_f1s  = [], []
    per_subject = []

    print(f"{'Subj':>5}  {'base_acc':>8}  {'ft_acc':>8}  {'delta':>7}  {'hard':>5}")
    print("─" * 50)

    for sid in subject_ids:
        tr_mask = (s_train == sid)
        va_mask = (s_val   == sid)
        te_mask = (s_test  == sid)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        train_ds = EMGDataset(X_train[tr_mask], y_train[tr_mask], s_train[tr_mask])
        val_ds   = EMGDataset(X_val[va_mask],   y_val[va_mask],   s_val[va_mask]) if va_mask.sum() else None
        test_ds  = EMGDataset(X_test[te_mask],  y_test[te_mask],  s_test[te_mask])

        train_loader = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=FT_BATCH, shuffle=False,
                                  num_workers=1, pin_memory=True) if val_ds else None
        test_loader  = DataLoader(test_ds,  batch_size=FT_BATCH, shuffle=False,
                                  num_workers=1, pin_memory=True)

        model = SubjectAwareCNNLSTMAttention(
            n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects
        ).to(DEVICE)
        model.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=True))

        _, base_acc, base_f1 = run_epoch(model, test_loader, criterion)
        base_accs.append(base_acc)
        base_f1s.append(base_f1)

        apply_freeze(model, FREEZE_MODULES)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR
        )

        best_val_f1  = -1.0
        patience_ctr = 0
        best_state   = None

        for epoch in range(1, FT_EPOCHS + 1):
            run_epoch(model, train_loader, criterion, optimizer)

            if val_loader is not None:
                _, val_acc, val_f1 = run_epoch(model, val_loader, criterion)
            else:
                _, val_acc, val_f1 = run_epoch(model, train_loader, criterion)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= FT_PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        _, ft_acc, ft_f1 = run_epoch(model, test_loader, criterion)
        ft_accs.append(ft_acc)
        ft_f1s.append(ft_f1)

        delta   = ft_acc - base_acc
        is_hard = sid in HARD_SUBJECTS

        per_subject.append({
            "subject_id": int(sid),
            "is_hard": is_hard,
            "base_acc": float(base_acc),
            "base_f1":  float(base_f1),
            "ft_acc":   float(ft_acc),
            "ft_f1":    float(ft_f1),
            "delta_acc": float(delta),
        })

        print(f"  s{sid:02d}  {base_acc:>8.4f}  {ft_acc:>8.4f}  "
              f"{delta:>+7.4f}  {'H' if is_hard else ' '}")

        del model, best_state
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    mean_base_acc = float(np.mean(base_accs))
    mean_ft_acc   = float(np.mean(ft_accs))
    mean_delta    = mean_ft_acc - mean_base_acc
    std_ft_acc    = float(np.std(ft_accs))

    hard_ft = [r["ft_acc"] for r in per_subject if r["is_hard"]]
    easy_ft = [r["ft_acc"] for r in per_subject if not r["is_hard"]]

    print("\n" + "═" * 50)
    print(f"[v33 최종 결과]")
    print(f"  v32 pooled base mean acc : {mean_base_acc*100:.2f}%  (재현 검증)")
    print(f"  v33 fine-tuned mean acc  : {mean_ft_acc*100:.2f}%  (std={std_ft_acc*100:.2f}p)")
    print(f"  delta                    : {mean_delta:+.4f}  ({mean_delta*100:+.2f}p)")
    if hard_ft:
        print(f"  hard subj acc  : {np.mean(hard_ft)*100:.2f}%  ({len(hard_ft)} 피험자)")
    if easy_ft:
        print(f"  easy subj acc  : {np.mean(easy_ft)*100:.2f}%  ({len(easy_ft)} 피험자)")
    print(f"  총 소요 시간 : {elapsed/60:.1f}분")
    print()
    print("비교:")
    print(f"  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)")
    print(f"  v32  68.70%  v28 전처리 + v8 arch (pooled)")
    print(f"  v31  68.38%  팀 10Hz + v8 arch (fine-tuned)")
    print(f"  v33  {mean_ft_acc*100:.2f}%  v28 전처리 + v8 arch (fine-tuned, {mean_delta*100:+.2f}p)")

    summary = {
        "model": "v33_finetuned",
        "base": "v32",
        "freeze_mode": "freeze_cnn",
        "ft_epochs": FT_EPOCHS,
        "ft_patience": FT_PATIENCE,
        "ft_lr": FT_LR,
        "ft_batch": FT_BATCH,
        "v32_base_mean_acc": mean_base_acc,
        "finetuned_mean_acc": mean_ft_acc,
        "finetuned_std_acc": std_ft_acc,
        "delta_acc": mean_delta,
        "hard_mean_acc": float(np.mean(hard_ft)) if hard_ft else None,
        "easy_mean_acc": float(np.mean(easy_ft)) if easy_ft else None,
        "elapsed_sec": elapsed,
        "baselines": {
            "v10_finetuned":  {"overall_accuracy": 0.6910},
            "v32_pooled":     {"overall_accuracy": 0.6870},
            "v31_finetuned":  {"overall_accuracy": 0.6838},
        },
        "per_subject": per_subject,
    }

    with open(os.path.join(OUT_DIR, "result_v33.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v33.json")


if __name__ == "__main__":
    main()
