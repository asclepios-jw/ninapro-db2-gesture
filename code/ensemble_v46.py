"""
v46: 학습 완료된 특징 모델들의 확률 평균 앙상블 (추론만, 재학습 없음).

동기: v44(TD)·v45(TD+FD) 특징 추가의 증분이 diminishing(각 ~+0.1p, hard 정체).
      이미 학습해둔 다양한 강한 pooled 모델을 앙상블해 headroom을 확인 + 뱅킹.

앙상블 대상 (pooled base, 4개):
  v44 원본   (TD arch,    seed 무)   results/v44/best_model_v44.pth
  v44 rep1   (TD arch,    seed 101)  results/v44_rep1/best_model_v44.pth
  v44 rep2   (TD arch,    seed 202)  results/v44_rep2/best_model_v44.pth
  v45        (TD+FD arch, seed 무)   results/v45/best_model_v45.pth

방법: 각 모델의 test softmax 확률을 구해 평균 → argmax. (확률 평균 앙상블)
비교 기준: 최고 단일 pooled = v45b 70.31%. (앙상블은 pooled 레벨 비교)

주의: 앙상블은 배포 시 추론 비용 ×N. 여기서는 headroom 확인 목적.
"""
import os, sys, json, time
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
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def build(arch, n_channels, n_classes, n_subjects):
    if arch == "v44":
        return SubjectAwareCNNLSTMAttentionFeat(n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects)
    return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs = []
    for xb, yb, sb in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        probs.append(F.softmax(model(xb, sb), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def metrics(probs, labels, subs):
    preds = probs.argmax(1)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)
    hidx = np.isin(subs, list(HARD_SUBJECTS))
    hard = accuracy_score(labels[hidx],  preds[hidx])
    easy = accuracy_score(labels[~hidx], preds[~hidx])
    return acc, f1, hard, easy


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}\n데이터 로드 중...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy"); y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy");  y_test  = np.load(f"{DATA_DIR}/y_test.npy")
    s_test  = np.load(f"{DATA_DIR}/s_test.npy")

    le = LabelEncoder(); le.fit(y_train)
    y_test_e = le.transform(y_test)
    n_classes = len(le.classes_); n_channels = X_test.shape[2]
    n_subjects = int(np.load(f"{DATA_DIR}/s_train.npy").max())

    test_loader = DataLoader(EMGDataset(X_test, y_test_e, s_test),
                             batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    labels = y_test_e; subs = s_test

    print(f"Test {len(X_test):,}  n_classes={n_classes}  멤버 {len(MEMBERS)}개\n")
    print(f"{'멤버':>10}  {'acc':>7}  {'macroF1':>7}  {'hard':>7}  {'easy':>7}")
    print("─" * 48)

    all_probs = []
    singles = []
    for name, path, arch in MEMBERS:
        model = build(arch, n_channels, n_classes, n_subjects).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        p = predict_probs(model, test_loader)
        all_probs.append(p)
        acc, f1, hard, easy = metrics(p, labels, subs)
        singles.append({"member": name, "acc": acc, "macro_f1": f1, "hard": hard, "easy": easy})
        print(f"{name:>10}  {acc*100:>6.2f}%  {f1:>7.4f}  {hard*100:>6.2f}%  {easy*100:>6.2f}%")
        del model; torch.cuda.empty_cache()

    def ens(idxs):
        avg = np.mean([all_probs[i] for i in idxs], axis=0)
        return metrics(avg, labels, subs)

    print("─" * 48)
    # 3× v44 (동일 arch) 앙상블
    a3, f3, h3, e3 = ens([0, 1, 2])
    print(f"{'ENS v44×3':>10}  {a3*100:>6.2f}%  {f3:>7.4f}  {h3*100:>6.2f}%  {e3*100:>6.2f}%")
    # 전체 4개 (TD×3 + TD+FD)
    a4, f4, h4, e4 = ens([0, 1, 2, 3])
    print(f"{'ENS all4':>10}  {a4*100:>6.2f}%  {f4:>7.4f}  {h4*100:>6.2f}%  {e4*100:>6.2f}%")

    best_single = max(s["acc"] for s in singles)
    elapsed = time.time() - t0

    print("\n" + "═" * 48)
    print("[v46 앙상블 결과 — pooled 레벨]")
    print(f"  최고 단일 pooled : {best_single*100:.2f}%  (v45b 70.31% 기준)")
    print(f"  앙상블 v44×3     : {a3*100:.2f}%  hard {h3*100:.2f}%  macroF1 {f3:.4f}")
    print(f"  앙상블 all4      : {a4*100:.2f}%  hard {h4*100:.2f}%  macroF1 {f4:.4f}")
    print(f"  vs 최고 단일     : v44×3 {(a3-best_single)*100:+.2f}p / all4 {(a4-best_single)*100:+.2f}p")
    print(f"  소요: {elapsed/60:.1f}분 (추론만)")
    print()
    print("참고: 앙상블은 pooled 레벨. 단일 pooled→adaptive는 v44/v45에서 +0.25~0.30p였음.")
    print("      배포 시 추론 비용 ×멤버수 trade-off.")

    result = {
        "model": "v46_ensemble",
        "members": [m[0] for m in MEMBERS],
        "singles": singles,
        "ensemble_v44x3": {"acc": a3, "macro_f1": f3, "hard": h3, "easy": e3},
        "ensemble_all4":  {"acc": a4, "macro_f1": f4, "hard": h4, "easy": e4},
        "best_single_pooled": best_single,
        "elapsed_sec": elapsed,
        "note": "pooled-level probability averaging; inference only",
    }
    with open(f"{OUT_DIR}/result_v46.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/result_v46.json")


if __name__ == "__main__":
    main()
