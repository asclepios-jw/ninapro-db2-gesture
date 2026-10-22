"""
v49: 사후 클래스 사전확률 보정(logit adjustment)으로 macro-F1 향상 (추론만).

배경: acc 0.73 vs macro-F1 0.64. 예측이 다수 클래스(rest 등)로 쏠려 희귀
      제스처를 못 맞힘. 사후 보정으로 재학습 없이 macro-F1을 산다.

방법 (Menon et al. 2020, logit adjustment):
  pred = argmax_c [ log p(c|x) − τ · log π(c) ]
  π(c) = train 클래스 빈도. τ=0(기존) → τ↑(균형).
  τ*는 검증셋(rep5) macro-F1 최대점으로 선택 후 test 적용 (test 과적합 방지).

대상:
  A) 앙상블 8모델 (v48 최적 조합)  ← 최고 모델
  B) 단일 v45                       ← 배포 관점 비교
출력: 각 대상의 τ 스윕 곡선(val), τ*에서 test acc/macro-F1, 베이스라인 대비.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8  import SubjectAwareCNNLSTMAttention
from models_v41 import SubjectAwareResCNNLSTMAttention
from models_v44 import SubjectAwareCNNLSTMAttentionFeat
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v49"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
R = "/mnt/data/medical_project/ninapro_db2_project/results"

# v48 최적 8개 조합
ENSEMBLE = [
    ("v44_orig", f"{R}/v44/best_model_v44.pth",      "v44"),
    ("v44_rep1", f"{R}/v44_rep1/best_model_v44.pth", "v44"),
    ("v44_rep2", f"{R}/v44_rep2/best_model_v44.pth", "v44"),
    ("v45",      f"{R}/v45/best_model_v45.pth",      "v45"),
    ("v45_r2",   f"{R}/v45_rep2/best_model_v45.pth", "v45"),
    ("v40",      f"{R}/v40/best_model_v40.pth",      "v8"),
    ("v41",      f"{R}/v41/best_model_v41.pth",      "v41"),
    ("v38",      f"{R}/v38/best_model_v38.pth",      "v8"),
]
SINGLE = ("v45", f"{R}/v45/best_model_v45.pth", "v45")


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64)); self.s = torch.from_numpy(s.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def build(key, nc, ncl, ns):
    if key == "v8":  return SubjectAwareCNNLSTMAttention(n_channels=nc, n_classes=ncl, n_subjects=ns)
    if key == "v41": return SubjectAwareResCNNLSTMAttention(n_channels=nc, n_classes=ncl, n_subjects=ns)
    if key == "v44": return SubjectAwareCNNLSTMAttentionFeat(n_channels=nc, n_classes=ncl, n_subjects=ns)
    return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc, n_classes=ncl, n_subjects=ns)


@torch.no_grad()
def probs_of(path, key, loader, nc, ncl, ns):
    m = build(key, nc, ncl, ns).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval(); out = []
    for xb, yb, sb in loader:
        out.append(F.softmax(m(xb.to(DEVICE), sb.to(DEVICE)), dim=1).cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def evaluate(pred, labels, subs):
    hidx = np.isin(subs, list(HARD_SUBJECTS))
    return (accuracy_score(labels, pred),
            f1_score(labels, pred, average="macro", zero_division=0),
            accuracy_score(labels[hidx], pred[hidx]))


def sweep(name, val_probs, vy, vs, test_probs, ty, ts, log_prior, taus):
    """val에서 τ* 선택 후 test 평가. 곡선도 반환."""
    logv = np.log(val_probs + 1e-9)
    logt = np.log(test_probs + 1e-9)
    curve = []
    for t in taus:
        vpred = (logv - t * log_prior).argmax(1)
        vacc, vf1, _ = evaluate(vpred, vy, vs)
        curve.append((t, vacc, vf1))
    # τ* = val macro-F1 최대
    tstar = max(curve, key=lambda r: r[2])[0]
    # baseline τ=0, τ*
    p0 = (logt - 0.0 * log_prior).argmax(1)
    ps = (logt - tstar * log_prior).argmax(1)
    a0, f0, h0 = evaluate(p0, ty, ts)
    a1, f1_, h1 = evaluate(ps, ty, ts)
    print(f"\n[{name}]  τ* (val macroF1 최대) = {tstar:.2f}")
    print(f"  τ=0 (기존)  : test acc {a0*100:.2f}%  macroF1 {f0:.4f}  hard {h0*100:.2f}%")
    print(f"  τ={tstar:.2f}    : test acc {a1*100:.2f}%  macroF1 {f1_:.4f}  hard {h1*100:.2f}%")
    print(f"  변화        : acc {(a1-a0)*100:+.2f}p  macroF1 {f1_-f0:+.4f}  hard {(h1-h0)*100:+.2f}p")
    # test 곡선 몇 점
    print(f"  test 곡선(τ: acc/macroF1): ", end="")
    for t in [0.0, 0.3, 0.6, 1.0]:
        pt = (logt - t * log_prior).argmax(1)
        at, ft, _ = evaluate(pt, ty, ts)
        print(f"{t:.1f}:{at*100:.1f}/{ft:.3f}  ", end="")
    print()
    return {"name": name, "tau_star": tstar,
            "base": {"acc": a0, "macro_f1": f0, "hard": h0},
            "corrected": {"acc": a1, "macro_f1": f1_, "hard": h1}}


def main():
    print(f"Device: {DEVICE}\n데이터 로드...")
    Xtr = np.load(f"{DATA_DIR}/X_train.npy"); ytr = np.load(f"{DATA_DIR}/y_train.npy")
    Xv  = np.load(f"{DATA_DIR}/X_val.npy");   yv  = np.load(f"{DATA_DIR}/y_val.npy");  sv = np.load(f"{DATA_DIR}/s_val.npy")
    Xte = np.load(f"{DATA_DIR}/X_test.npy");  yte = np.load(f"{DATA_DIR}/y_test.npy"); ste = np.load(f"{DATA_DIR}/s_test.npy")
    le = LabelEncoder(); le.fit(ytr)
    ytr_e = le.transform(ytr); vy = le.transform(yv); ty = le.transform(yte)
    ncl = len(le.classes_); nc = Xte.shape[2]; ns = int(np.load(f"{DATA_DIR}/s_train.npy").max())

    # 클래스 사전확률 (train 빈도)
    counts = np.bincount(ytr_e, minlength=ncl).astype(np.float64)
    prior = counts / counts.sum()
    log_prior = np.log(prior + 1e-9)[None, :]   # (1, ncl)
    print(f"클래스 {ncl}개. 최대 빈도(rest 등) {prior.max()*100:.1f}% / 최소 {prior.min()*100:.2f}%")

    val_loader  = DataLoader(EMGDataset(Xv,  vy, sv),  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(EMGDataset(Xte, ty, ste), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    taus = np.round(np.linspace(0, 1.5, 16), 3)

    # A) 앙상블 8모델
    print("\n앙상블 멤버 확률 계산(val+test)...")
    ev = np.mean([probs_of(p, k, val_loader,  nc, ncl, ns) for _, p, k in ENSEMBLE], axis=0)
    et = np.mean([probs_of(p, k, test_loader, nc, ncl, ns) for _, p, k in ENSEMBLE], axis=0)
    res_ens = sweep("앙상블 8모델", ev, vy, sv, et, ty, ste, log_prior, taus)

    # B) 단일 v45
    print("\n단일 v45 확률 계산(val+test)...")
    sv_ = probs_of(SINGLE[1], SINGLE[2], val_loader,  nc, ncl, ns)
    st_ = probs_of(SINGLE[1], SINGLE[2], test_loader, nc, ncl, ns)
    res_single = sweep("단일 v45", sv_, vy, sv, st_, ty, ste, log_prior, taus)

    with open(f"{OUT_DIR}/result_v49.json", "w") as f:
        json.dump({"model": "v49_prior_correction", "ensemble": res_ens, "single": res_single,
                   "taus": taus.tolist()}, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v49.json")


if __name__ == "__main__":
    main()
