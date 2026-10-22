"""
v38: v36 (증강 + focal) 장기 학습 — 100 epochs.

목적: v36이 50ep을 조기 종료 없이 완주하고 ep49까지 상승 중이었음(미수렴).
      에포크를 100으로 늘려 pooled base의 상한을 끌어올린다.

v36 대비 변경 (세 가지, 함께 조정):
  N_EPOCHS 50 → 100
  CosineAnnealingLR T_max 50 → 100  (스케줄 동반 조정 필수:
     안 바꾸면 50ep 이후 LR이 바닥에 붙어 학습이 멈춤)
  PATIENCE 8 → 15  (100ep 기준 더 느린 수렴 허용)

나머지 완전 동일: v28 전처리, v8 arch, 증강(jitter/magwarp/timewarp/mixup),
                 Focal Loss(γ=2), AdamW 1e-3, batch=256.

비교 기준:
  v36  69.25%  증강+focal, 50ep (base)
  v37  69.83%  v36 + adaptive fine-tuning
  v38     ?    증강+focal, 100ep  → 이후 fine-tuning(v39 예정) 재적용 대상

데이터: 전처리전달_v28/
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v38"
CKPT     = os.path.join(OUT_DIR, "best_model_v38.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 100   # v36: 50
PATIENCE     = 15    # v36: 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX        = 100   # v36: 50  (N_EPOCHS와 일치시켜야 함)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}

# ── 증강 하이퍼파라미터 (v36과 동일) ─────────────────────────────────────────
JITTER_P,   JITTER_SIGMA  = 0.5, 0.06
MAGWARP_P,  MAGWARP_SIGMA = 0.5, 0.20
TIMEWARP_P, TIMEWARP_SIGMA= 0.5, 0.15
WARP_KNOTS  = 4
MIXUP_P,    MIXUP_ALPHA   = 0.5, 0.20
FOCAL_GAMMA = 2.0


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


def aug_jitter(x, sigma):
    return x + torch.randn_like(x) * sigma


def aug_magnitude_warp(x, sigma, knots):
    B, C, T = x.shape
    Cpair = C // 2
    knot_vals = torch.randn(B, Cpair, knots, device=x.device) * sigma + 1.0
    curve = F.interpolate(knot_vals, size=T, mode="linear", align_corners=True)
    curve = torch.cat([curve, curve], dim=1)
    return x * curve


def aug_time_warp(x, sigma, knots):
    B, C, T = x.shape
    speed = torch.exp(torch.randn(B, 1, knots, device=x.device) * sigma)
    speed = F.interpolate(speed, size=T, mode="linear", align_corners=True)
    cum   = torch.cumsum(speed, dim=2)
    cum   = cum - cum[:, :, :1]
    cum   = cum / cum[:, :, -1:]
    grid_x = cum * 2.0 - 1.0
    grid = torch.zeros(B, 1, T, 2, device=x.device)
    grid[..., 0] = grid_x
    grid[..., 1] = 0.0
    x4  = x.unsqueeze(2)
    out = F.grid_sample(x4, grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out.squeeze(2)


def apply_augment(x):
    if torch.rand(1).item() < MAGWARP_P:
        x = aug_magnitude_warp(x, MAGWARP_SIGMA, WARP_KNOTS)
    if torch.rand(1).item() < TIMEWARP_P:
        x = aug_time_warp(x, TIMEWARP_SIGMA, WARP_KNOTS)
    if torch.rand(1).item() < JITTER_P:
        x = aug_jitter(x, JITTER_SIGMA)
    return x


def mixup_batch(x, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, idx, lam


def train_epoch(model, loader, optimizer):
    model.train()
    preds, labels = [], []
    for xb, yb, sb in loader:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        xb = apply_augment(xb)

        use_mixup = torch.rand(1).item() < MIXUP_P
        if use_mixup:
            xb, idx, lam = mixup_batch(xb, MIXUP_ALPHA)
            yb2 = yb[idx]

        logits = model(xb, sb)
        if use_mixup:
            loss = lam * focal_loss(logits, yb, FOCAL_GAMMA) \
                 + (1.0 - lam) * focal_loss(logits, yb2, FOCAL_GAMMA)
        else:
            loss = focal_loss(logits, yb, FOCAL_GAMMA)

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        preds.extend(logits.detach().argmax(1).cpu().numpy())
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
    hard_idx = np.isin(subs, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(labels[hard_idx],  preds[hard_idx])
    easy_acc = accuracy_score(labels[~hard_idx], preds[~hard_idx])
    return acc, macro_f1, hard_acc, easy_acc


def main():
    print(f"Device: {DEVICE}")
    print(f"장기 학습: {N_EPOCHS}ep, T_max={T_MAX}, patience={PATIENCE}")
    print(f"증강+focal(γ={FOCAL_GAMMA}) — v36과 동일")
    t0 = time.time()

    print("\n데이터 로드 중...")
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

    train_loader = DataLoader(EMGDataset(X_train, y_train, s_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EMGDataset(X_val,   y_val,   s_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(EMGDataset(X_test,  y_test,  s_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SubjectAwareCNNLSTMAttention(
        n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"아키텍처: SubjectAwareCNNLSTMAttention (v8)  파라미터: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=1e-6
    )

    best_val_f1  = -1.0
    best_epoch   = 0
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
            best_epoch  = epoch
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

    print(f"\n[테스트 평가]  best epoch = {best_epoch}")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    test_acc, test_f1, test_hard, test_easy = evaluate(model, test_loader)
    gap     = test_easy - test_hard
    elapsed = time.time() - t0

    print(f"Overall acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Macro-F1     : {test_f1:.4f}")
    print(f"Hard acc     : {test_hard:.4f}  ({test_hard*100:.2f}%)")
    print(f"Easy acc     : {test_easy:.4f}  ({test_easy*100:.2f}%)")
    print(f"Gap          : {gap:.4f}  ({gap*100:.2f}p)")
    print(f"\n비교:")
    print(f"  v36  69.25%  증강+focal, 50ep")
    print(f"  v37  69.83%  v36 + adaptive fine-tuning")
    print(f"  v38  {test_acc*100:.2f}%  증강+focal, {N_EPOCHS}ep (best ep{best_epoch})")
    print(f"  vs v36: {(test_acc-0.6925)*100:+.2f}p")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v38_v36_long100ep",
        "architecture": "SubjectAwareCNNLSTMAttention (models_v8.py)",
        "data": "전처리전달_v28",
        "n_epochs": N_EPOCHS, "t_max": T_MAX, "patience": PATIENCE,
        "best_epoch": best_epoch,
        "augment": {
            "jitter":   {"p": JITTER_P,   "sigma": JITTER_SIGMA},
            "mag_warp": {"p": MAGWARP_P,  "sigma": MAGWARP_SIGMA, "knots": WARP_KNOTS},
            "time_warp":{"p": TIMEWARP_P, "sigma": TIMEWARP_SIGMA, "knots": WARP_KNOTS},
            "mixup":    {"p": MIXUP_P,    "alpha": MIXUP_ALPHA},
        },
        "loss": f"focal(gamma={FOCAL_GAMMA})",
        "overall_accuracy": test_acc,
        "overall_macro_f1": test_f1,
        "hard_acc": test_hard,
        "easy_acc": test_easy,
        "gap": gap,
        "best_val_macro_f1": best_val_f1,
        "elapsed_sec": elapsed,
        "baselines": {
            "v36_50ep":      {"overall_accuracy": 0.6925},
            "v37_finetuned": {"overall_accuracy": 0.6983},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v38.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v38.json")


if __name__ == "__main__":
    main()
