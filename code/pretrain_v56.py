"""
v56 (1/2): 자기지도 대조학습(self-supervised contrastive) 사전학습.

동기: 후처리(평활/분할/학습보정, v52~55)는 천장에 근접했고, 코어 프레임 모델
      (앙상블 73%)이 진짜 병목. 특히 hard subject의 표현이 약함. 라벨 없이 신호
      구조로 CNN 백본을 먼저 학습(대조학습)시켜 표현을 강화하면, 이후 지도학습
      finetune에서 코어 프레임 성능(및 hard)을 올릴 수 있는지 검증한다.

방법 (SimCLR/NT-Xent):
  - 인코더 = models_v8의 CNN 백본(동일 구조) + 투영헤드(MLP→128d).
  - 각 윈도우에 라벨보존 증강(magwarp/timewarp/jitter)을 2번 독립 적용 → 2개 뷰.
  - 같은 윈도우의 두 뷰는 positive, 배치 내 다른 윈도우는 negative. NT-Xent.
  - 학습셋 = X_train(rep1~4, 라벨 미사용). CNN state_dict만 저장.
  주의: 전극shift(v43 실패)는 제외 — 라벨 의미 파괴. 증강은 label-preserving만.

출력: results/v56/pretrained_cnn_v56.pth  (→ train_cnn_v56.py가 로드해 finetune)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v56"
os.makedirs(OUT_DIR, exist_ok=True)
CNN_CKPT = os.path.join(OUT_DIR, "pretrained_cnn_v56.pth")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 하이퍼파라미터 ───────────────────────────────────────────────────────────
EPOCHS       = 25
BATCH        = 512          # NT-Xent은 배치(=negative 수)가 클수록 유리
SAMPLES_PER_EP = 160_000    # 매 epoch 무작위 부분집합 (속도/RAM)
LR           = 2e-3
TEMP         = 0.2          # NT-Xent 온도
PROJ_DIM     = 128
# 증강(train_cnn_v40와 동일 강도, label-preserving만)
JITTER_SIGMA  = 0.06
MAGWARP_SIGMA = 0.30
TIMEWARP_SIGMA= 0.25
WARP_KNOTS    = 4


def make_cnn(n_channels=24):
    # models_v8.SubjectAwareCNNLSTMAttention.cnn 과 완전 동일 구조 (state_dict 호환)
    return nn.Sequential(
        nn.Conv1d(n_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ELU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ELU(), nn.MaxPool1d(2),
    )


class Encoder(nn.Module):
    def __init__(self, n_channels=24, proj_dim=PROJ_DIM):
        super().__init__()
        self.cnn = make_cnn(n_channels)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ELU(), nn.Linear(128, proj_dim))
    def forward(self, x):
        h = self.cnn(x)              # (B,128,150)
        h = h.mean(dim=2)            # global avg pool → (B,128)
        z = self.proj(h)
        return F.normalize(z, dim=1)


# ── 증강 (label-preserving; v40에서 가져옴) ──────────────────────────────────
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
    cum = torch.cumsum(speed, dim=2); cum = cum - cum[:, :, :1]; cum = cum / cum[:, :, -1:]
    grid_x = cum * 2.0 - 1.0
    grid = torch.zeros(B, 1, T, 2, device=x.device); grid[..., 0] = grid_x
    out = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", align_corners=True, padding_mode="border")
    return out.squeeze(2)

def two_views(x):
    def one(v):
        v = aug_magnitude_warp(v, MAGWARP_SIGMA, WARP_KNOTS)
        v = aug_time_warp(v, TIMEWARP_SIGMA, WARP_KNOTS)
        v = aug_jitter(v, JITTER_SIGMA)
        return v
    return one(x.clone()), one(x.clone())


def nt_xent(z1, z2, temp=TEMP):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2B,D)
    sim = (z @ z.t()) / temp                        # (2B,2B)
    self_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(self_mask, float("-inf"))
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}  | SSL 대조학습(NT-Xent) | batch={BATCH} temp={TEMP} epochs={EPOCHS}")
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"), mmap_mode="r")   # (N,600,24)
    N = X.shape[0]; nch = X.shape[2]
    print(f"train windows: {N:,}  channels: {nch}  (라벨 미사용)")

    enc = Encoder(n_channels=nch).to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    hist = []
    for ep in range(1, EPOCHS + 1):
        enc.train()
        sel = np.random.choice(N, size=min(SAMPLES_PER_EP, N), replace=False)
        sel.sort()                                   # mmap 순차 접근 유리
        np.random.shuffle(sel)
        losses = []
        for a in range(0, len(sel), BATCH):
            idx = sel[a:a + BATCH]
            if len(idx) < 8: continue
            xb = np.asarray(X[np.sort(idx)]).transpose(0, 2, 1).astype(np.float32)  # (B,24,600)
            xb = torch.from_numpy(xb).to(DEVICE, non_blocking=True)
            v1, v2 = two_views(xb)
            z1, z2 = enc(v1), enc(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        m = float(np.mean(losses))
        hist.append({"epoch": ep, "nt_xent": m})
        print(f"  ep {ep:>2}/{EPOCHS}  NT-Xent {m:.4f}  ({(time.time()-t0)/60:.1f}min)")

    torch.save(enc.cnn.state_dict(), CNN_CKPT)       # CNN 백본만 저장(finetune에서 로드)
    json.dump({"model": "v56_ssl_pretrain", "epochs": EPOCHS, "batch": BATCH,
               "temp": TEMP, "samples_per_ep": SAMPLES_PER_EP, "history": hist,
               "elapsed_sec": time.time() - t0},
              open(os.path.join(OUT_DIR, "pretrain_v56.json"), "w"), indent=2)
    print(f"\n저장: {CNN_CKPT}")
    print(f"총 {(time.time()-t0)/60:.1f}분 → 다음: train_cnn_v56.py (사전학습 CNN 로드 후 finetune)")


if __name__ == "__main__":
    main()
