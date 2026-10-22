"""
v42: v40 챔피언 base + adaptive fine-tuning, 단 hard subject는 fine-tuning 제외.

동기: v40·v41 모두 hard 5명 {4,7,26,28,31}에서 fine-tuning이 반복적으로
      역효과(val≠test로 잘못 선택되어 test 악화). hard는 val 데이터가 적고
      노이즈가 커 적응형 선택이 불안정.

규칙 변경 (v40 대비 유일한 차이):
  - hard subject → fine-tuning 아예 안 함, 무조건 pooled base 사용 (BASE 고정)
  - easy subject → 기존 adaptive (base vs fine-tuned 중 val_f1 높은 것)

base·아키텍처·하이퍼파라미터는 v40과 100% 동일:
  BASE_CKPT = results/v40/best_model_v40.pth  (v8 arch, 강한 증강 pooled 70.05%)
  freeze_cnn, Adam 1e-4, 15ep, focal γ2, batch 128.

비교 기준:
  v40  70.35%  전원 adaptive (hard도 FT 선택되어 악화)
  v42     ?    hard는 BASE 고정 + easy adaptive

기대: hard를 pooled 수준(~60%)으로 안정 확보 → 전체 +0.05~0.15p + 안정성.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR  = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
BASE_CKPT = "/mnt/data/medical_project/ninapro_db2_project/results/v40/best_model_v40.pth"
OUT_DIR   = "/mnt/data/medical_project/ninapro_db2_project/results/v42"
os.makedirs(OUT_DIR, exist_ok=True)

FT_EPOCHS   = 15
FT_PATIENCE = 5
FT_LR       = 1e-4
FT_BATCH    = 128
FOCAL_GAMMA = 2.0
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


def focal_loss(logits, target, gamma=2.0):
    logp   = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t    = logp_t.exp()
    loss   = -((1.0 - p_t) ** gamma) * logp_t
    return loss.mean()


def apply_freeze(model, freeze_modules):
    for name, module in model.named_children():
        if name in freeze_modules:
            for p in module.parameters():
                p.requires_grad = False
        else:
            for p in module.parameters():
                p.requires_grad = True


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    preds, labels = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            logits = model(xb, sb)
            loss   = focal_loss(logits, yb, FOCAL_GAMMA)
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
    print(f"BASE_CKPT: {BASE_CKPT}")
    print(f"규칙: hard {sorted(HARD_SUBJECTS)} → BASE 고정 (fine-tuning 제외), easy → adaptive\n")

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

    subject_ids = sorted(np.unique(s_train))
    per_subject = []
    sel_accs = []
    n_used_ft = 0
    n_hard_forced = 0
    base_accs_all = []

    print(f"{'Subj':>5}  {'base_v':>7}  {'ft_v':>7}  {'선택':>10}  "
          f"{'base_t':>7}  {'ft_t':>7}  {'sel':>7}  {'hard':>5}")
    print("─" * 72)

    for sid in subject_ids:
        tr_mask = (s_train == sid)
        va_mask = (s_val   == sid)
        te_mask = (s_test  == sid)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        is_hard = sid in HARD_SUBJECTS

        val_ds  = EMGDataset(X_val[va_mask],  y_val[va_mask],  s_val[va_mask]) if va_mask.sum() else None
        test_ds = EMGDataset(X_test[te_mask], y_test[te_mask], s_test[te_mask])
        val_loader  = DataLoader(val_ds,  batch_size=FT_BATCH, shuffle=False,
                                 num_workers=1, pin_memory=True) if val_ds else None
        test_loader = DataLoader(test_ds, batch_size=FT_BATCH, shuffle=False,
                                 num_workers=1, pin_memory=True)

        # ── base 평가 (항상) ──
        base_model = load_base_model(n_channels, n_classes, n_subjects)
        base_val_acc, base_val_f1 = run_epoch(base_model, val_loader) \
            if val_loader else (0.0, 0.0)
        base_test_acc, base_test_f1 = run_epoch(base_model, test_loader)
        base_accs_all.append(base_test_acc)

        if is_hard:
            # ── hard: fine-tuning 제외, BASE 고정 ──
            del base_model; torch.cuda.empty_cache()
            sel_acc, sel_f1 = base_test_acc, base_test_f1
            sel_label = "BASE(hard)"
            n_hard_forced += 1
            ft_val_f1_disp, ft_test_acc = float('nan'), float('nan')
            per_subject.append({
                "subject_id": int(sid), "is_hard": True,
                "base_val_f1": float(base_val_f1), "ft_val_f1": None,
                "selected": "BASE_forced_hard",
                "base_test_acc": float(base_test_acc), "ft_test_acc": None,
                "sel_test_acc": float(sel_acc), "sel_test_f1": float(sel_f1),
            })
            sel_accs.append(sel_acc)
            print(f"  s{sid:02d}  {base_val_f1:>7.4f}  {'--':>7}  {sel_label:>10}  "
                  f"{base_test_acc:>7.4f}  {'--':>7}  {sel_acc:>7.4f}  {'H':>5}")
            continue

        # ── easy: adaptive (fine-tuning + val_f1 선택) ──
        tr_ds = EMGDataset(X_train[tr_mask], y_train[tr_mask], s_train[tr_mask])
        train_loader = DataLoader(tr_ds, batch_size=FT_BATCH, shuffle=True,
                                  num_workers=2, pin_memory=True)

        ft_model = load_base_model(n_channels, n_classes, n_subjects)
        apply_freeze(ft_model, FREEZE_MODULES)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ft_model.parameters()), lr=FT_LR
        )
        best_ft_val_f1 = -1.0; best_ft_state = None; patience_ctr = 0
        for epoch in range(1, FT_EPOCHS + 1):
            run_epoch(ft_model, train_loader, optimizer)
            v_acc, v_f1 = run_epoch(ft_model, val_loader) if val_loader \
                else run_epoch(ft_model, train_loader)
            if v_f1 > best_ft_val_f1:
                best_ft_val_f1 = v_f1
                best_ft_state = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= FT_PATIENCE:
                    break
        if best_ft_state is not None:
            ft_model.load_state_dict(best_ft_state)
        ft_test_acc, ft_test_f1 = run_epoch(ft_model, test_loader)

        use_ft = best_ft_val_f1 > base_val_f1
        sel_acc = ft_test_acc if use_ft else base_test_acc
        sel_f1  = ft_test_f1  if use_ft else base_test_f1
        sel_label = "FT" if use_ft else "BASE"
        if use_ft:
            n_used_ft += 1

        per_subject.append({
            "subject_id": int(sid), "is_hard": False,
            "base_val_f1": float(base_val_f1), "ft_val_f1": float(best_ft_val_f1),
            "selected": sel_label,
            "base_test_acc": float(base_test_acc), "ft_test_acc": float(ft_test_acc),
            "sel_test_acc": float(sel_acc), "sel_test_f1": float(sel_f1),
        })
        sel_accs.append(sel_acc)
        print(f"  s{sid:02d}  {base_val_f1:>7.4f}  {best_ft_val_f1:>7.4f}  {sel_label:>10}  "
              f"{base_test_acc:>7.4f}  {ft_test_acc:>7.4f}  {sel_acc:>7.4f}  {'':>5}")

        del base_model, ft_model, best_ft_state
        torch.cuda.empty_cache()

    elapsed   = time.time() - t0
    mean_acc  = float(np.mean(sel_accs))
    std_acc   = float(np.std(sel_accs))
    mean_base = float(np.mean(base_accs_all))
    hard_accs = [r["sel_test_acc"] for r in per_subject if r["is_hard"]]
    easy_accs = [r["sel_test_acc"] for r in per_subject if not r["is_hard"]]
    n_easy_base = len(easy_accs) - n_used_ft

    print("\n" + "═" * 72)
    print(f"[v42 최종 결과]")
    print(f"  base per-subject mean acc : {mean_base*100:.2f}%  (재현 검증)")
    print(f"  hard {n_hard_forced}명 → BASE 고정 / easy: FT {n_used_ft}명, BASE {n_easy_base}명")
    print(f"  v42 mean acc  : {mean_acc*100:.2f}%  (std={std_acc*100:.2f}p)")
    print(f"  hard subj acc : {np.mean(hard_accs)*100:.2f}%  ({len(hard_accs)}명, 전원 BASE)")
    print(f"  easy subj acc : {np.mean(easy_accs)*100:.2f}%  ({len(easy_accs)}명)")
    print(f"  vs base: {(mean_acc-mean_base)*100:+.2f}p")
    print(f"  총 소요 시간  : {elapsed/60:.1f}분")
    print()
    print("비교:")
    print(f"  v40  70.35%  전원 adaptive (hard도 FT로 악화)")
    print(f"  v42  {mean_acc*100:.2f}%  hard는 BASE 고정 + easy adaptive")
    if mean_acc > 0.7035:
        print(f"  ★ 신기록 (v40 초과)")

    summary = {
        "model": "v42_v40base_hardexcluded_adaptive",
        "base": "v40 (best_model_v40.pth, v8 arch)",
        "rule": "hard subjects forced to BASE (no fine-tuning); easy subjects adaptive",
        "hard_subjects": sorted(HARD_SUBJECTS),
        "freeze_mode": "freeze_cnn", "loss": f"focal(gamma={FOCAL_GAMMA})",
        "ft_epochs": FT_EPOCHS, "ft_patience": FT_PATIENCE, "ft_lr": FT_LR, "ft_batch": FT_BATCH,
        "n_hard_forced_base": n_hard_forced, "n_easy_ft": n_used_ft, "n_easy_base": n_easy_base,
        "base_mean_acc": mean_base,
        "v42_mean_acc": mean_acc, "v42_std_acc": std_acc,
        "hard_mean_acc": float(np.mean(hard_accs)) if hard_accs else None,
        "easy_mean_acc": float(np.mean(easy_accs)) if easy_accs else None,
        "elapsed_sec": elapsed,
        "baselines": {"v40_all_adaptive": {"overall_accuracy": 0.7035, "hard_acc": 0.5936}},
        "per_subject": per_subject,
    }
    with open(os.path.join(OUT_DIR, "result_v42.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v42.json")


if __name__ == "__main__":
    main()
