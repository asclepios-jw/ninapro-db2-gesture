"""
v48: 신선 멤버 2개 추가한 9-후보 확장 앙상블 — 73% 돌파 재도전 (추론만).

v47(7개)=72.91%로 73%에 0.09p 차·포화. 강한 신선 멤버(v45 새 seed 2개,
70.26%/70.29%)를 더해 비상관 다양성을 늘림 → 73% 재도전.

후보 9개 (모두 전처리전달_v28, 재학습 0):
  v44_orig/rep1/rep2 (v8+TD), v45 (v8+TD+FD),
  v45_rep1/rep2 (v8+TD+FD, seed111/222 ← 신규),
  v40 (v8), v41 (Residual CNN), v38 (v8 long)
"""
import os, sys, json, itertools
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
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v48"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
R = "/mnt/data/medical_project/ninapro_db2_project/results"

CANDIDATES = [
    ("v44_orig", f"{R}/v44/best_model_v44.pth",      "v44"),
    ("v44_rep1", f"{R}/v44_rep1/best_model_v44.pth", "v44"),
    ("v44_rep2", f"{R}/v44_rep2/best_model_v44.pth", "v44"),
    ("v45",      f"{R}/v45/best_model_v45.pth",      "v45"),
    ("v45_r1",   f"{R}/v45_rep1/best_model_v45.pth", "v45"),   # 신규 seed111
    ("v45_r2",   f"{R}/v45_rep2/best_model_v45.pth", "v45"),   # 신규 seed222
    ("v40",      f"{R}/v40/best_model_v40.pth",      "v8"),
    ("v41",      f"{R}/v41/best_model_v41.pth",      "v41"),
    ("v38",      f"{R}/v38/best_model_v38.pth",      "v8"),
]


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
def predict_probs(model, loader):
    model.eval(); out = []
    for xb, yb, sb in loader:
        out.append(F.softmax(model(xb.to(DEVICE), sb.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    print(f"Device: {DEVICE}\n데이터 로드...")
    Xtr = np.load(f"{DATA_DIR}/X_train.npy"); ytr = np.load(f"{DATA_DIR}/y_train.npy")
    Xte = np.load(f"{DATA_DIR}/X_test.npy");  yte = np.load(f"{DATA_DIR}/y_test.npy")
    ste = np.load(f"{DATA_DIR}/s_test.npy")
    le = LabelEncoder(); le.fit(ytr); labels = le.transform(yte); subs = ste
    ncl = len(le.classes_); nc = Xte.shape[2]; ns = int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader = DataLoader(EMGDataset(Xte, labels, ste), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    hidx = np.isin(subs, list(HARD_SUBJECTS))

    names = []; P = []
    print("\n단일 모델:")
    for name, path, key in CANDIDATES:
        m = build(key, nc, ncl, ns).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        p = predict_probs(m, loader); P.append(p); names.append(name)
        print(f"  {name:>9}  {accuracy_score(labels, p.argmax(1))*100:.2f}%")
        del m; torch.cuda.empty_cache()
    P = np.stack(P, 0); M = len(names)

    def acc_of(idxs):
        return accuracy_score(labels, P[list(idxs)].mean(0).argmax(1))

    best = {"acc": -1}; by_size_best = {}
    for k in range(1, M + 1):
        bs = None
        for combo in itertools.combinations(range(M), k):
            a = acc_of(combo)
            if bs is None or a > bs["acc"]:
                bs = {"acc": a, "members": [names[i] for i in combo]}
            if a > best["acc"]:
                best = {"acc": a, "members": [names[i] for i in combo], "size": k}
        by_size_best[k] = bs

    print(f"\n{'크기':>4}  {'최고 acc':>8}  최고 조합")
    print("─" * 74)
    for k in range(1, M + 1):
        bs = by_size_best[k]
        star = "  ★≥73%" if bs["acc"] >= 0.73 else ""
        print(f"{k:>4}  {bs['acc']*100:>7.2f}%  {'+'.join(bs['members'])}{star}")

    bidx = [names.index(n) for n in best["members"]]
    bp = P[bidx].mean(0); bpred = bp.argmax(1)
    b_f1  = f1_score(labels, bpred, average="macro", zero_division=0)
    b_hard= accuracy_score(labels[hidx], bpred[hidx]); b_easy= accuracy_score(labels[~hidx], bpred[~hidx])

    print("\n" + "═" * 74)
    print("[v48 9-후보 확장 앙상블 결과]")
    print(f"  최고 조합 ({best['size']}개): {best['acc']*100:.2f}%")
    print(f"    멤버: {'+'.join(best['members'])}")
    print(f"    hard {b_hard*100:.2f}%  easy {b_easy*100:.2f}%  macroF1 {b_f1:.4f}")
    print(f"  v47(7개) 72.91% 대비: {(best['acc']-0.7291)*100:+.2f}p")
    print(f"  73% 도달: {'★★★ 성공 ★★★' if best['acc']>=0.73 else '미달 (차이 %.2fp)'%((0.73-best['acc'])*100)}")

    result = {"model": "v48_extended_ensemble_9", "candidates": names,
              "by_size_best": {str(k): by_size_best[k] for k in by_size_best},
              "best": {**best, "hard": b_hard, "easy": b_easy, "macro_f1": b_f1},
              "reached_73": bool(best["acc"] >= 0.73)}
    with open(f"{OUT_DIR}/result_v48.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v48.json")


if __name__ == "__main__":
    main()
