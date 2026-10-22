"""
v36: v28 전처리 + v8 arch + EMG 특화 데이터 증강 + Focal Loss (pooled).

목적: v32(68.70%) baseline 위에 두 가지만 추가하여 순수 효과 측정:
  (1) EMG 특화 온라인 데이터 증강 (GPU, 학습 중에만)
  (2) label smoothing → Focal Loss(γ=2) 교체

동기:
  - v32의 train_acc(0.88) vs val_acc(0.70) = 18p 격차 → 명백한 과적합.
    증강은 과적합 모델에 거의 항상 도움 (일반화 개선).
  - v34 oversampling 실패 = hard subject 문제는 "양"이 아니라 "다양성".
    단순 복제가 아닌 실제 신호 변형으로 공략.
  - macro-F1(0.57) << acc(0.69) → 소수 클래스/hard 부실 → Focal Loss로 집중.

증강 (Um et al. 2017 계열, 근전도 신호에 표준적):
  - Jitter          : 가우시안 노이즈 주입
  - Magnitude warp  : 채널별 부드러운 스케일 곡선 (EMG ch와 대응 envelope ch 연동)
  - Time warp       : 부드러운 시간축 왜곡 (grid_sample, 단조 warp)
  - Mixup           : 같은 배치 내 윈도우 선형 결합

비교 기준:
  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)
  v35  68.99%  v28 전처리 + v8 arch (adaptive fine-tuned)
  v32  68.70%  v28 전처리 + v8 arch (pooled, 증강 없음)  ← 직접 비교 대상
  v36     ?    v28 전처리 + v8 arch (pooled + 증강 + focal)

데이터: 전처리전달_v28/ (이미 정규화 완료, std≈0.85)
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
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v36"
CKPT     = os.path.join(OUT_DIR, "best_model_v36.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}   # 1-indexed

# ── 증강 하이퍼파라미터 (보수적 시작값) ──────────────────────────────────────
JITTER_P,   JITTER_SIGMA  = 0.5, 0.06   # data std≈0.85 → 약 7% 노이즈
MAGWARP_P,  MAGWARP_SIGMA = 0.5, 0.20
TIMEWARP_P, TIMEWARP_SIGMA= 0.5, 0.15
WARP_KNOTS  = 4
MIXUP_P,    MIXUP_ALPHA   = 0.5, 0.20
FOCAL_GAMMA = 2.0


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))  # (N, 24, 600)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))   # 1-indexed

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


# ── Focal Loss ──────────────────────────────────────────────────────────────
def focal_loss(logits, target, gamma=2.0):
    """다중클래스 focal loss (mean 축약). label smoothing 대체."""
    logp   = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # 정답 클래스 log-prob
    p_t    = logp_t.exp()
    loss   = -((1.0 - p_t) ** gamma) * logp_t
    return loss.mean()


# ── GPU 온라인 증강 (배치 텐서 x: (B, C, T)) ─────────────────────────────────
def aug_jitter(x, sigma):
    return x + torch.randn_like(x) * sigma


def aug_magnitude_warp(x, sigma, knots):
    """채널별 부드러운 스케일 곡선. EMG ch c 와 envelope ch c+12 를 동일 곡선으로 연동."""
    B, C, T = x.shape
    Cpair = C // 2   # 12 (EMG=[0:12], envelope=[12:24])
    knot_vals = torch.randn(B, Cpair, knots, device=x.device) * sigma + 1.0
    curve = F.interpolate(knot_vals, size=T, mode="linear", align_corners=True)  # (B, Cpair, T)
    curve = torch.cat([curve, curve], dim=1)  # (B, C, T)  ← env를 emg와 동일 스케일
    return x * curve


def aug_time_warp(x, sigma, knots):
    """부드러운 단조 시간축 왜곡 (grid_sample). 양수 speed의 누적합으로 단조성 보장."""
    B, C, T = x.shape
    speed = torch.exp(torch.randn(B, 1, knots, device=x.device) * sigma)  # 양수
    speed = F.interpolate(speed, size=T, mode="linear", align_corners=True)  # (B, 1, T)
    cum   = torch.cumsum(speed, dim=2)
    cum   = cum - cum[:, :, :1]        # 0 시작
    cum   = cum / cum[:, :, -1:]       # [0, 1] 정규화 (단조 증가)
    grid_x = cum * 2.0 - 1.0           # [-1, 1]
    grid = torch.zeros(B, 1, T, 2, device=x.device)
    grid[..., 0] = grid_x              # x(=시간) 좌표
    grid[..., 1] = 0.0                 # y 좌표 (H=1)
    x4  = x.unsqueeze(2)               # (B, C, 1, T)
    out = F.grid_sample(x4, grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out.squeeze(2)              # (B, C, T)


def apply_augment(x):
    """배치 단위 확률 게이팅으로 각 증강 적용 (곡선/노이즈는 샘플별 독립)."""
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

        # ── 증강 ──
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
        # tr_acc는 원본 라벨 yb 기준 근사치 (mixup 시 참고용)
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
    print(f"증강: jitter(p={JITTER_P},σ={JITTER_SIGMA}) "
          f"magwarp(p={MAGWARP_P},σ={MAGWARP_SIGMA}) "
          f"timewarp(p={TIMEWARP_P},σ={TIMEWARP_SIGMA}) "
          f"mixup(p={MIXUP_P},α={MIXUP_ALPHA})")
    print(f"손실: Focal Loss (γ={FOCAL_GAMMA})")
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
    n_subjects = int(s_train.max())   # 40 (1-indexed)

    print(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}")
    print(f"X_train mean={X_train.mean():.4f}  std={X_train.std():.4f}\n")

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
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )

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
    print(f"\n비교:")
    print(f"  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)")
    print(f"  v35  68.99%  v28 전처리 + v8 arch (adaptive fine-tuned)")
    print(f"  v32  68.70%  v28 전처리 + v8 arch (pooled, 증강 없음)")
    print(f"  v36  {test_acc*100:.2f}%  v28 전처리 + v8 arch (pooled + 증강 + focal)")
    print(f"  vs v32: overall {(test_acc-0.6870)*100:+.2f}p / "
          f"hard {(test_hard-0.5784)*100:+.2f}p / macro-F1 {(test_f1-0.5718):+.4f}")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v36_v28preproc_v8arch_augment_focal",
        "architecture": "SubjectAwareCNNLSTMAttention (models_v8.py)",
        "data": "전처리전달_v28",
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
            "v10_finetuned":  {"overall_accuracy": 0.6910},
            "v35_adaptive":   {"overall_accuracy": 0.6899},
            "v32_pooled":     {"overall_accuracy": 0.6870,
                               "hard_acc": 0.5784, "macro_f1": 0.5718},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v36.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v36.json")


if __name__ == "__main__":
    main()
