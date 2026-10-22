"""
v50: v45(TD+FD) + 온건한 클래스 균형 샘플러 → macro-F1 향상 시도 (pooled).

진단(v49): 낮은 macro-F1의 원인은 12개 제스처가 rest로 새는 낮은 recall.
           rest가 train의 44%로 지배 → 모델이 rest로 기울어짐.
처방: WeightedRandomSampler(sqrt-역빈도)로 rest를 온건히 다운샘플 →
      제스처 윈도우를 상대적으로 더 자주 학습 → 제스처 recall↑ → macro-F1↑.

주의(v3~v6 교훈): 손실 보정(focal)에 강한 class weight를 겹치면 rest 과억제로
      폭락. 그래서 focal은 그대로 두고 '샘플러 하나만' 온건하게(sqrt) 바꿈.

v45 대비 유일한 변경: train 샘플러 (아키텍처·증강·손실 동일).
평가에 rest_recall / gesture_recall 추가 → 트레이드오프 직접 확인.
데이터: 전처리전달_v28/
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v50"
CKPT     = os.path.join(OUT_DIR, "best_model_v50.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX        = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
BALANCE_POWER = 0.5   # 0=균형없음, 0.5=sqrt(온건), 1=완전역빈도(공격적)

JITTER_P, JITTER_SIGMA   = 0.5, 0.06
MAGWARP_P, MAGWARP_SIGMA = 0.5, 0.30
TIMEWARP_P, TIMEWARP_SIGMA = 0.5, 0.25
WARP_KNOTS = 4
MIXUP_P, MIXUP_ALPHA = 0.5, 0.30
FOCAL_GAMMA = 2.0


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64)); self.s = torch.from_numpy(s.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def focal_loss(logits, target, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return (-((1.0 - logp_t.exp()) ** gamma) * logp_t).mean()


def aug_jitter(x, s): return x + torch.randn_like(x) * s
def aug_magnitude_warp(x, sigma, knots):
    B, C, T = x.shape; Cp = C // 2
    kv = torch.randn(B, Cp, knots, device=x.device) * sigma + 1.0
    curve = F.interpolate(kv, size=T, mode="linear", align_corners=True)
    return x * torch.cat([curve, curve], dim=1)
def aug_time_warp(x, sigma, knots):
    B, C, T = x.shape
    sp = torch.exp(torch.randn(B, 1, knots, device=x.device) * sigma)
    sp = F.interpolate(sp, size=T, mode="linear", align_corners=True)
    cum = torch.cumsum(sp, 2); cum = cum - cum[:, :, :1]; cum = cum / cum[:, :, -1:]
    grid = torch.zeros(B, 1, T, 2, device=x.device); grid[..., 0] = cum * 2 - 1
    return F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", align_corners=True, padding_mode="border").squeeze(2)
def apply_augment(x):
    if torch.rand(1).item() < MAGWARP_P:  x = aug_magnitude_warp(x, MAGWARP_SIGMA, WARP_KNOTS)
    if torch.rand(1).item() < TIMEWARP_P: x = aug_time_warp(x, TIMEWARP_SIGMA, WARP_KNOTS)
    if torch.rand(1).item() < JITTER_P:   x = aug_jitter(x, JITTER_SIGMA)
    return x
def mixup_batch(x, alpha):
    lam = float(np.random.beta(alpha, alpha)); idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], idx, lam


def train_epoch(model, loader, opt):
    model.train(); preds, labels = [], []
    for xb, yb, sb in loader:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        xb = apply_augment(xb)
        um = torch.rand(1).item() < MIXUP_P
        if um: xb, idx, lam = mixup_batch(xb, MIXUP_ALPHA); yb2 = yb[idx]
        logits = model(xb, sb)
        loss = (lam * focal_loss(logits, yb, FOCAL_GAMMA) + (1 - lam) * focal_loss(logits, yb2, FOCAL_GAMMA)) if um \
               else focal_loss(logits, yb, FOCAL_GAMMA)
        opt.zero_grad(); loss.backward(); opt.step()
        preds.extend(logits.detach().argmax(1).cpu().numpy()); labels.extend(yb.cpu().numpy())
    return accuracy_score(labels, preds)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, labels, subs = [], [], []
    for xb, yb, sb in loader:
        preds.extend(model(xb.to(DEVICE), sb.to(DEVICE)).argmax(1).cpu().numpy())
        labels.extend(yb.numpy()); subs.extend(sb.numpy())
    preds, labels, subs = np.array(preds), np.array(labels), np.array(subs)
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    hidx = np.isin(subs, list(HARD_SUBJECTS))
    hard = accuracy_score(labels[hidx], preds[hidx]); easy = accuracy_score(labels[~hidx], preds[~hidx])
    # rest(인코딩 0) recall vs 제스처(1..) 평균 recall
    rec = recall_score(labels, preds, labels=list(range(50)), average=None, zero_division=0)
    rest_rec = rec[0]; gest_rec = rec[1:].mean()
    return acc, mf1, hard, easy, rest_rec, gest_rec


def main():
    print(f"Device: {DEVICE}")
    print(f"v50: v45(TD+FD) + 클래스균형 샘플러(sqrt, power={BALANCE_POWER})")
    t0 = time.time()
    print("\n데이터 로드...")
    Xtr = np.load(f"{DATA_DIR}/X_train.npy"); ytr = np.load(f"{DATA_DIR}/y_train.npy"); str_ = np.load(f"{DATA_DIR}/s_train.npy")
    Xv = np.load(f"{DATA_DIR}/X_val.npy"); yv = np.load(f"{DATA_DIR}/y_val.npy"); sv = np.load(f"{DATA_DIR}/s_val.npy")
    Xte = np.load(f"{DATA_DIR}/X_test.npy"); yte = np.load(f"{DATA_DIR}/y_test.npy"); ste = np.load(f"{DATA_DIR}/s_test.npy")
    le = LabelEncoder(); le.fit(ytr)
    ytr = le.transform(ytr); yv = le.transform(yv); yte = le.transform(yte)
    ncl = len(le.classes_); nc = Xtr.shape[2]; ns = int(str_.max())

    # 클래스 균형 샘플러
    cls_count = np.bincount(ytr, minlength=ncl).astype(np.float64)
    w_class = 1.0 / np.power(cls_count + 1.0, BALANCE_POWER)
    sample_w = w_class[ytr]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(), num_samples=len(ytr), replacement=True)
    rest_frac = cls_count[0] / cls_count.sum()
    eff_rest = (w_class[0] * cls_count[0]) / (w_class * cls_count).sum()
    print(f"rest(클래스0) 원본 비중 {rest_frac*100:.1f}% → 샘플러 유효 비중 {eff_rest*100:.1f}%  (온건 완화)")
    print(f"Train {len(Xtr):,}  Val {len(Xv):,}  Test {len(Xte):,}  n_classes={ncl}\n")

    train_loader = DataLoader(EMGDataset(Xtr, ytr, str_), batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EMGDataset(Xv, yv, sv), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(EMGDataset(Xte, yte, ste), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc, n_classes=ncl, n_subjects=ns).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=1e-6)

    best_f1 = -1; best_ep = 0; pc = 0; hist = []
    print(f"{'Ep':>3}  {'tr_acc':>7}  {'val_acc':>7}  {'val_f1':>7}  {'restR':>6}  {'gestR':>6}")
    for ep in range(1, N_EPOCHS + 1):
        tr = train_epoch(model, train_loader, opt)
        va, vf1, vh, ve, vrr, vgr = evaluate(model, val_loader); sch.step()
        flag = ""
        if vf1 > best_f1:
            best_f1 = vf1; best_ep = ep; torch.save(model.state_dict(), CKPT); pc = 0; flag = " ↑"
        else:
            pc += 1
            if pc >= PATIENCE: print(f"Early stop ep{ep}"); break
        print(f"{ep:>3}  {tr:>7.4f}  {va:>7.4f}  {vf1:>7.4f}  {vrr:>6.3f}  {vgr:>6.3f}{flag}")
        hist.append({"epoch": ep, "val_acc": va, "val_macro_f1": vf1, "val_rest_rec": vrr, "val_gest_rec": vgr})

    print(f"\n[테스트]  best epoch={best_ep}")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    acc, mf1, hard, easy, rr, gr = evaluate(model, test_loader)
    el = time.time() - t0
    print(f"Overall acc : {acc*100:.2f}%")
    print(f"Macro-F1    : {mf1:.4f}")
    print(f"Hard/Easy   : {hard*100:.2f}% / {easy*100:.2f}%")
    print(f"rest recall : {rr:.3f}   gesture recall(1..49 평균): {gr:.3f}")
    print(f"\n비교 (pooled 단일):")
    print(f"  v45  acc 70.31%  macroF1 0.6002  (균형 없음)")
    print(f"  v50  acc {acc*100:.2f}%  macroF1 {mf1:.4f}  (sqrt 균형)")
    print(f"  변화: acc {(acc-0.7031)*100:+.2f}p  macroF1 {mf1-0.6002:+.4f}")
    print(f"총 시간: {el/60:.1f}분")

    json.dump({"model": "v50_balanced_sampler", "balance_power": BALANCE_POWER,
               "overall_accuracy": acc, "overall_macro_f1": mf1, "hard_acc": hard, "easy_acc": easy,
               "rest_recall": rr, "gesture_recall": gr, "best_epoch": best_ep, "best_val_f1": best_f1,
               "elapsed_sec": el, "baseline_v45": {"acc": 0.7031, "macro_f1": 0.6002}, "history": hist},
              open(f"{OUT_DIR}/result_v50.json", "w"), indent=2)
    print(f"저장: {OUT_DIR}/result_v50.json")


if __name__ == "__main__":
    main()
