"""
v35: v32 체크포인트 기반 adaptive per-subject fine-tuning.

전략: 피험자별로 val_f1 기준 fine-tuned vs base 중 나은 모델을 선택.
      v33에서 17개 피험자가 fine-tuning 후 악화됐는데,
      그 피험자들은 v32 base를 그대로 쓰면 손실을 회수할 수 있음.

선택 기준:
  ft_val_f1 > base_val_f1 → fine-tuned 모델로 test 평가
  ft_val_f1 ≤ base_val_f1 → base(v32) 모델로 test 평가

비교 기준:
  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)
  v33  68.89%  v28 전처리 + v8 arch (fine-tuned, 무조건)
  v32  68.70%  v28 전처리 + v8 arch (pooled)
  v35     ?    v28 전처리 + v8 arch (adaptive selection)

기대: 68.89% + 0.10~0.15p = 69.0% 내외

데이터: 전처리전달_v28/
BASE_CKPT: results/v32/best_model_v32.pth
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
OUT_DIR   = "/mnt/data/medical_project/ninapro_db2_project/results/v35"
os.makedirs(OUT_DIR, exist_ok=True)

FT_EPOCHS   = 15
FT_PATIENCE = 5
FT_LR       = 1e-4
FT_BATCH    = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}

FREEZE_MODULES = ["cnn"]


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))

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
    training = optimizer is not None
    model.train() if training else model.eval()
    preds, labels = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            loss   = criterion(logits, yb)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds.extend(logits.detach().argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def load_base_model(n_channels, n_classes, n_subjects):
    model = SubjectAwareCNNLSTMAttention(
        n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects
    ).to(DEVICE)
    model.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=True))
    return model


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
    n_subjects = int(s_train.max())

    print(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}\n")

    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    subject_ids  = sorted(np.unique(s_train))

    per_subject  = []
    ft_accs, ft_f1s = [], []
    n_used_ft    = 0   # fine-tuned 모델 선택 횟수

    print(f"{'Subj':>5}  {'base_v':>7}  {'ft_v':>7}  {'선택':>6}  "
          f"{'test_acc':>9}  {'delta_v33':>9}  {'hard':>5}")
    print("─" * 65)

    # v33 per-subject 결과 로드 (delta 비교용)
    v33_path = "/mnt/data/medical_project/ninapro_db2_project/results/v33/result_v33.json"
    v33_by_sid = {}
    if os.path.exists(v33_path):
        with open(v33_path) as f:
            v33_data = json.load(f)
        for r in v33_data.get("per_subject", []):
            v33_by_sid[r["subject_id"]] = r["ft_acc"]

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

        # ── base 모델 val/test 평가 ───────────────────────────────────────────
        base_model = load_base_model(n_channels, n_classes, n_subjects)
        base_val_acc, base_val_f1 = run_epoch(base_model, val_loader, criterion) \
            if val_loader else run_epoch(base_model, train_loader, criterion)
        base_test_acc, base_test_f1 = run_epoch(base_model, test_loader, criterion)

        # ── fine-tuning ───────────────────────────────────────────────────────
        ft_model = load_base_model(n_channels, n_classes, n_subjects)
        apply_freeze(ft_model, FREEZE_MODULES)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ft_model.parameters()), lr=FT_LR
        )

        best_ft_val_f1 = -1.0
        best_ft_state  = None
        patience_ctr   = 0

        for epoch in range(1, FT_EPOCHS + 1):
            run_epoch(ft_model, train_loader, criterion, optimizer)
            val_acc, val_f1 = run_epoch(ft_model, val_loader, criterion) \
                if val_loader else run_epoch(ft_model, train_loader, criterion)
            if val_f1 > best_ft_val_f1:
                best_ft_val_f1 = val_f1
                best_ft_state  = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}
                patience_ctr   = 0
            else:
                patience_ctr += 1
                if patience_ctr >= FT_PATIENCE:
                    break

        if best_ft_state is not None:
            ft_model.load_state_dict(best_ft_state)
        ft_test_acc, ft_test_f1 = run_epoch(ft_model, test_loader, criterion)

        # ── adaptive 선택: val_f1 기준 ────────────────────────────────────────
        use_ft   = best_ft_val_f1 > base_val_f1
        sel_acc  = ft_test_acc  if use_ft else base_test_acc
        sel_f1   = ft_test_f1   if use_ft else base_test_f1
        sel_label = "FT" if use_ft else "BASE"
        if use_ft:
            n_used_ft += 1

        is_hard = sid in HARD_SUBJECTS
        v33_acc = v33_by_sid.get(sid, None)
        delta_v33 = (sel_acc - v33_acc) if v33_acc is not None else float("nan")

        per_subject.append({
            "subject_id": int(sid),
            "is_hard": is_hard,
            "base_val_f1":  float(base_val_f1),
            "ft_val_f1":    float(best_ft_val_f1),
            "selected":     sel_label,
            "base_test_acc": float(base_test_acc),
            "ft_test_acc":  float(ft_test_acc),
            "sel_test_acc": float(sel_acc),
            "sel_test_f1":  float(sel_f1),
            "delta_vs_v33": float(delta_v33),
        })

        ft_accs.append(sel_acc)
        ft_f1s.append(sel_f1)

        print(f"  s{sid:02d}  {base_val_f1:>7.4f}  {best_ft_val_f1:>7.4f}  "
              f"{sel_label:>6}  {sel_acc:>9.4f}  "
              f"{delta_v33:>+9.4f}  {'H' if is_hard else ' '}")

        del base_model, ft_model, best_ft_state
        torch.cuda.empty_cache()

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    elapsed       = time.time() - t0
    mean_acc      = float(np.mean(ft_accs))
    std_acc       = float(np.std(ft_accs))
    hard_accs     = [r["sel_test_acc"] for r in per_subject if r["is_hard"]]
    easy_accs     = [r["sel_test_acc"] for r in per_subject if not r["is_hard"]]
    n_used_base   = len(per_subject) - n_used_ft

    print("\n" + "═" * 65)
    print(f"[v35 최종 결과]")
    print(f"  선택 통계: FT 사용 {n_used_ft}명 / BASE 사용 {n_used_base}명")
    print(f"  v35 adaptive mean acc : {mean_acc*100:.2f}%  (std={std_acc*100:.2f}p)")
    print(f"  hard subj acc : {np.mean(hard_accs)*100:.2f}%  ({len(hard_accs)} 피험자)")
    print(f"  easy subj acc : {np.mean(easy_accs)*100:.2f}%  ({len(easy_accs)} 피험자)")
    print(f"  총 소요 시간  : {elapsed/60:.1f}분")
    print()
    print("비교:")
    print(f"  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)")
    print(f"  v33  68.89%  v28 전처리 + v8 arch (무조건 fine-tuned)")
    print(f"  v32  68.70%  v28 전처리 + v8 arch (pooled)")
    print(f"  v35  {mean_acc*100:.2f}%  v28 전처리 + v8 arch (adaptive selection)")

    summary = {
        "model": "v35_adaptive_finetuned",
        "base": "v32",
        "freeze_mode": "freeze_cnn",
        "selection_criterion": "val_macro_f1: ft > base → use ft, else use base",
        "ft_epochs": FT_EPOCHS,
        "ft_patience": FT_PATIENCE,
        "ft_lr": FT_LR,
        "ft_batch": FT_BATCH,
        "n_subjects_ft": n_used_ft,
        "n_subjects_base": n_used_base,
        "adaptive_mean_acc": mean_acc,
        "adaptive_std_acc": std_acc,
        "hard_mean_acc": float(np.mean(hard_accs)) if hard_accs else None,
        "easy_mean_acc": float(np.mean(easy_accs)) if easy_accs else None,
        "elapsed_sec": elapsed,
        "baselines": {
            "v10_finetuned": {"overall_accuracy": 0.6910},
            "v33_finetuned": {"overall_accuracy": 0.6889},
            "v32_pooled":    {"overall_accuracy": 0.6870},
        },
        "per_subject": per_subject,
    }

    with open(os.path.join(OUT_DIR, "result_v35.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v35.json")


if __name__ == "__main__":
    main()
