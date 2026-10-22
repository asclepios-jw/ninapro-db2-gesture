"""
v43 (1/2): v40 파이프라인 + 전극 shift 증강.

목적: 피험자 간 변동의 큰 원인인 '전극 착용 회전'을 증강으로 모사해
      회전 불변성을 학습 → hard subject/gap 공략, 전원 동일 적용(라벨 불필요).

v40 대비 유일한 추가: 전극 shift 증강 (나머지 증강·손실·아키텍처 동일).

전극 shift 설계 (DB2, Atzori et al. 2014):
  · 채널 1–8 = 전완 둘레 등간격 8전극 → 링(회전 대상)
  · 채널 9–12 = 굴근/신근/이두/삼두 → 특수 위치(고정)
  · 24채널 = EMG[0:12] + ENV[12:24], ENV[i]는 EMG[i] 파생
  → 링 EMG(0–7)와 대응 포락선(12–19)을 '같은 양 k칸' 순환 이동,
    특수 전극(8–11, 20–23) 고정.
  · k ∈ {−1,+1} (1칸=45°), 샘플별 랜덤, 확률 0.5.
  ⚠️ .mat 채널 순서가 '앞 8개=둘레 링'이라는 표준 가정. RING_* 상수로 조정 가능.

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
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v43"
CKPT     = os.path.join(OUT_DIR, "best_model_v43.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX        = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}

# ── 증강 (v40과 동일, 강한 강도) ─────────────────────────────────────────────
JITTER_P,   JITTER_SIGMA  = 0.5, 0.06
MAGWARP_P,  MAGWARP_SIGMA = 0.5, 0.30
TIMEWARP_P, TIMEWARP_SIGMA= 0.5, 0.25
WARP_KNOTS  = 4
MIXUP_P,    MIXUP_ALPHA   = 0.5, 0.30
FOCAL_GAMMA = 2.0

# ── 전극 shift (v43 신규) ────────────────────────────────────────────────────
RING_EMG_START, RING_N = 0, 8      # EMG 링: 채널 0–7 (전완 둘레 8전극)
RING_ENV_START         = 12        # 대응 포락선: 채널 12–19
ELEC_SHIFT_P    = 0.5
ELEC_SHIFT_SET  = (-1, 1)          # ±1칸(45°), 샘플별 랜덤 방향


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


def aug_electrode_shift(x):
    """링 8전극(EMG 0–7 + 대응 포락선 12–19)을 샘플별로 ±1칸 순환 이동. 특수전극 고정."""
    B, C, T = x.shape
    dev = x.device
    ks_idx = torch.randint(0, len(ELEC_SHIFT_SET), (B,), device=dev)
    ks = torch.tensor(ELEC_SHIFT_SET, device=dev)[ks_idx]         # (B,) 샘플별 이동량
    idx = torch.arange(C, device=dev).unsqueeze(0).expand(B, C).clone()  # (B, C) 항등
    ring = torch.arange(RING_N, device=dev).unsqueeze(0)         # (1, 8) 출력 위치
    rolled = (ring - ks.unsqueeze(1)) % RING_N                   # (B, 8) 소스=(p-k)%8
    idx[:, RING_EMG_START:RING_EMG_START+RING_N] = RING_EMG_START + rolled
    idx[:, RING_ENV_START:RING_ENV_START+RING_N] = RING_ENV_START + rolled  # 동일 k
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, C, T))


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
    if torch.rand(1).item() < ELEC_SHIFT_P:          # v43 신규 (구조적 op 먼저)
        x = aug_electrode_shift(x)
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
    preds  = np.array(preds); labels = np.array(labels); subs = np.array(subs)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    hard_idx = np.isin(subs, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(labels[hard_idx],  preds[hard_idx])
    easy_acc = accuracy_score(labels[~hard_idx], preds[~hard_idx])
    return acc, macro_f1, hard_acc, easy_acc


def main():
    print(f"Device: {DEVICE}")
    print(f"아키텍처: SubjectAwareCNNLSTMAttention (v8, v40과 동일)")
    print(f"증강: v40 강한 증강 + 전극 shift(링 8전극 ±1칸, p={ELEC_SHIFT_P})")
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
    print(f"파라미터 수: {n_params:,}")

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
    print(f"\n비교 (pooled base):")
    print(f"  v40b  70.05%  강한 증강 (전극 shift 없음)  hard 59.54%  gap 12.00p")
    print(f"  v43b  {test_acc*100:.2f}%  + 전극 shift        hard {test_hard*100:.2f}%  gap {gap*100:.2f}p")
    print(f"  vs v40b: {(test_acc-0.7005)*100:+.2f}p")
    print(f"총 소요 시간: {elapsed/60:.1f}분")
    print(f"\n→ 다음: finetune_v43.py로 adaptive fine-tuning 적용")

    result = {
        "model": "v43_pooled_electrode_shift",
        "stage": "pooled_base",
        "architecture": "SubjectAwareCNNLSTMAttention (models_v8.py, v40과 동일)",
        "data": "전처리전달_v28",
        "n_epochs": N_EPOCHS, "t_max": T_MAX, "best_epoch": best_epoch,
        "augment": {
            "jitter":   {"p": JITTER_P,   "sigma": JITTER_SIGMA},
            "mag_warp": {"p": MAGWARP_P,  "sigma": MAGWARP_SIGMA, "knots": WARP_KNOTS},
            "time_warp":{"p": TIMEWARP_P, "sigma": TIMEWARP_SIGMA, "knots": WARP_KNOTS},
            "mixup":    {"p": MIXUP_P,    "alpha": MIXUP_ALPHA},
            "electrode_shift": {"p": ELEC_SHIFT_P, "shift_set": list(ELEC_SHIFT_SET),
                                "ring": "EMG 0-7 + ENV 12-19"},
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
            "v40_pooled": {"overall_accuracy": 0.7005, "hard_acc": 0.5954, "gap": 0.1200},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v43_pooled.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v43_pooled.json")


if __name__ == "__main__":
    main()
