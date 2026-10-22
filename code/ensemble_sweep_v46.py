"""
v46 앙상블 특성 파악 — 멤버 수·조합 sweep (추론만).

목적: +2.01p 앙상블 이득이 (a)특정 4개의 우연인지 일반적인지, (b)몇 개에서
      포화되는지(배포 추론비용 결정)를 확인.

방법: 4개 멤버의 test 확률을 1회 계산 후, 모든 비어있지 않은 부분집합(15개)을
      확률 평균으로 평가. 크기(1~4)별로 acc 분포(min~max, 평균) 집계.
"""
import os, sys, json, itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v44 import SubjectAwareCNNLSTMAttentionFeat
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v46"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
R = "/mnt/data/medical_project/ninapro_db2_project/results"

MEMBERS = [
    ("v44_orig", f"{R}/v44/best_model_v44.pth",      "v44"),
    ("v44_rep1", f"{R}/v44_rep1/best_model_v44.pth", "v44"),
    ("v44_rep2", f"{R}/v44_rep2/best_model_v44.pth", "v44"),
    ("v45",      f"{R}/v45/best_model_v45.pth",      "v45"),
]


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64)); self.s = torch.from_numpy(s.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def build(arch, nc, ncl, ns):
    return (SubjectAwareCNNLSTMAttentionFeat if arch == "v44" else SubjectAwareCNNLSTMAttentionFeatFD)(
        n_channels=nc, n_classes=ncl, n_subjects=ns)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval(); out = []
    for xb, yb, sb in loader:
        out.append(F.softmax(model(xb.to(DEVICE), sb.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out, 0)


def met(probs, labels, subs):
    p = probs.argmax(1)
    hidx = np.isin(subs, list(HARD_SUBJECTS))
    return (accuracy_score(labels, p),
            f1_score(labels, p, average="macro", zero_division=0),
            accuracy_score(labels[hidx], p[hidx]),
            accuracy_score(labels[~hidx], p[~hidx]))


def main():
    print(f"Device: {DEVICE}\n데이터 로드...")
    Xtr = np.load(f"{DATA_DIR}/X_train.npy"); ytr = np.load(f"{DATA_DIR}/y_train.npy")
    Xte = np.load(f"{DATA_DIR}/X_test.npy");  yte = np.load(f"{DATA_DIR}/y_test.npy")
    ste = np.load(f"{DATA_DIR}/s_test.npy")
    le = LabelEncoder(); le.fit(ytr); yte_e = le.transform(yte)
    ncl = len(le.classes_); nc = Xte.shape[2]; ns = int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader = DataLoader(EMGDataset(Xte, yte_e, ste), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    labels, subs = yte_e, ste

    # 각 멤버 확률 1회 계산
    probs = []
    for name, path, arch in MEMBERS:
        m = build(arch, nc, ncl, ns).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        probs.append(predict_probs(m, loader)); del m; torch.cuda.empty_cache()
    names = [m[0] for m in MEMBERS]

    # 모든 부분집합 평가
    n = len(MEMBERS)
    by_size = {k: [] for k in range(1, n + 1)}
    rows = []
    for k in range(1, n + 1):
        for combo in itertools.combinations(range(n), k):
            avg = np.mean([probs[i] for i in combo], axis=0)
            acc, f1, hard, easy = met(avg, labels, subs)
            by_size[k].append(acc)
            rows.append({"size": k, "members": [names[i] for i in combo],
                         "acc": acc, "macro_f1": f1, "hard": hard, "easy": easy})

    # 출력: 크기별 요약
    print(f"\n{'크기':>4}  {'조합수':>5}  {'acc min':>8}  {'acc mean':>9}  {'acc max':>8}  {'최고 조합':<28}")
    print("─" * 78)
    for k in range(1, n + 1):
        accs = by_size[k]
        best = max([r for r in rows if r["size"] == k], key=lambda r: r["acc"])
        print(f"{k:>4}  {len(accs):>5}  {min(accs)*100:>7.2f}%  {np.mean(accs)*100:>8.2f}%  "
              f"{max(accs)*100:>7.2f}%  {'+'.join(best['members']):<28}")

    # 최고 전체
    best_all = max(rows, key=lambda r: r["acc"])
    single_best = max(r["acc"] for r in rows if r["size"] == 1)
    print("\n" + "═" * 78)
    print("[해석]")
    print(f"  최고 단일: {single_best*100:.2f}%")
    for k in range(2, n + 1):
        mn, mx = min(by_size[k]), max(by_size[k])
        print(f"  {k}개 앙상블: {mn*100:.2f}~{mx*100:.2f}%  (최고 조합 +{(mx-single_best)*100:.2f}p, "
              f"최악 조합도 +{(mn-single_best)*100:.2f}p)")
    print(f"  전체 최고: {best_all['acc']*100:.2f}%  hard {best_all['hard']*100:.2f}%  "
          f"macroF1 {best_all['macro_f1']:.4f}  ({'+'.join(best_all['members'])})")
    # 포화 판단: 3개 최고 vs 4개
    best3 = max(by_size[3]); best4 = max(by_size[4])
    print(f"  포화 확인: 3개 최고 {best3*100:.2f}% → 4개 {best4*100:.2f}% (증분 +{(best4-best3)*100:.2f}p)")

    with open(f"{OUT_DIR}/result_v46_sweep.json", "w") as f:
        json.dump({"members": names, "by_size_acc": {str(k): by_size[k] for k in by_size},
                   "all_subsets": rows, "best": best_all, "single_best": single_best}, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v46_sweep.json")


if __name__ == "__main__":
    main()
