"""
v52: 온라인 시간 평활 — oracle 분할 없이 스트리밍 프레임 성능 회복 (추론 후처리).

v51: 프레임(73%) ↔ 반복 다수결(88%)의 15p 격차 = 시간적 착시. 그러나 88%는
     제스처 경계를 안다는 oracle 전제. 이 버전은 '분할 정보 없이 인과적/온라인'
     시간 평활만으로 격차를 얼마나 회수하는지(정직한 배포 수치) 측정.

방법: 앙상블 8모델 윈도우 확률을 피험자 내 시간순으로 두고,
  - 인과적(causal): 각 시점 i에서 직전 W개 윈도우 확률 평균 → argmax (실시간).
  - 중심(centered): i 기준 ±W/2 윈도우 평균 → argmax (약간의 지연 허용).
  경계는 피험자 내에서만, 제스처 경계는 '모름'(그게 정직한 조건).
  W를 스윕해 acc/macro-F1 곡선. 상한 참조 = v51 oracle 반복 다수결(88.04/0.846).
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
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v52"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
STRIDE_MS = 150   # 윈도우 stride (컨텍스트 ms 환산용)
R = "/mnt/data/medical_project/ninapro_db2_project/results"
ENSEMBLE = [
    (f"{R}/v44/best_model_v44.pth","v44"),(f"{R}/v44_rep1/best_model_v44.pth","v44"),
    (f"{R}/v44_rep2/best_model_v44.pth","v44"),(f"{R}/v45/best_model_v45.pth","v45"),
    (f"{R}/v45_rep2/best_model_v45.pth","v45"),(f"{R}/v40/best_model_v40.pth","v8"),
    (f"{R}/v41/best_model_v41.pth","v41"),(f"{R}/v38/best_model_v38.pth","v8"),
]


class DS(Dataset):
    def __init__(s, X, y, sub):
        s.X=torch.from_numpy(X.transpose(0,2,1).astype(np.float32)); s.y=torch.from_numpy(y.astype(np.int64)); s.s=torch.from_numpy(sub.astype(np.int64))
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.X[i], s.y[i], s.s[i]


def build(k,nc,ncl,ns):
    if k=="v8":  return SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v41": return SubjectAwareResCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v44": return SubjectAwareCNNLSTMAttentionFeat(n_channels=nc,n_classes=ncl,n_subjects=ns)
    return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc,n_classes=ncl,n_subjects=ns)


@torch.no_grad()
def probs(path,k,loader,nc,ncl,ns):
    m=build(k,nc,ncl,ns).to(DEVICE); m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True)); m.eval()
    o=[]
    for xb,yb,sb in loader: o.append(F.softmax(m(xb.to(DEVICE),sb.to(DEVICE)),1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(o,0)


def smooth_preds(P, subs, W, mode):
    """피험자 내에서 시간 평활. mode='causal'(직전 W) 또는 'centered'(±W/2). → 예측 배열."""
    N, C = P.shape
    out = np.empty(N, dtype=np.int64)
    for s in np.unique(subs):
        idx = np.where(subs == s)[0]          # 저장순서=시간순 (연속 블록)
        seg = P[idx]                          # (L, C)
        L = len(idx)
        cum = np.vstack([np.zeros((1, C)), np.cumsum(seg, axis=0)])  # (L+1, C)
        for j in range(L):
            if mode == "causal":
                a = max(0, j - W + 1); b = j
            else:  # centered
                h = W // 2; a = max(0, j - h); b = min(L - 1, j + h)
            ssum = cum[b + 1] - cum[a]
            out[idx[j]] = ssum.argmax()
        # (벡터화 대신 명료성 우선; N~10만이라 충분히 빠름)
    return out


def met(y, p, subs, ncl):
    hidx = np.isin(subs, list(HARD_SUBJECTS))
    return (accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0, labels=range(ncl)),
            accuracy_score(y[hidx], p[hidx]))


def main():
    print("데이터 로드 (test rep6, 시간순)...")
    ytr=np.load(f"{DATA_DIR}/y_train.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); y=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader=DataLoader(DS(Xte,y,ste),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    print(f"앙상블 8모델 윈도우 확률 계산 (윈도우 {len(y):,}개)...")
    P=np.mean([probs(p,k,loader,nc,ncl,ns) for p,k in ENSEMBLE],axis=0)

    base_acc, base_f1, base_hard = met(y, P.argmax(1), ste, ncl)
    print(f"\n[기준] 윈도우 레벨(W=1): acc {base_acc*100:.2f}%  macroF1 {base_f1:.4f}  hard {base_hard*100:.2f}%")
    print(f"[상한 참조] v51 oracle 반복 다수결: acc 88.04%  macroF1 0.8457  hard 83.88%\n")

    Ws = [1, 3, 5, 7, 9, 13, 19, 27, 39]
    print(f"{'W':>3} {'ms':>5} │ {'causal acc':>10} {'macroF1':>8} {'hard':>6} │ {'center acc':>10} {'macroF1':>8} {'hard':>6}")
    print("─"*76)
    rows = []
    for W in Ws:
        ms = (W - 1) * STRIDE_MS + 300   # 대략적 컨텍스트 길이(첫 창 300ms + stride*(W-1))
        ca, cf, ch = met(y, smooth_preds(P, ste, W, "causal"), ste, ncl)
        na, nf, nh = met(y, smooth_preds(P, ste, W, "centered"), ste, ncl)
        print(f"{W:>3} {ms:>5} │ {ca*100:>9.2f}% {cf:>8.4f} {ch*100:>5.1f}% │ {na*100:>9.2f}% {nf:>8.4f} {nh*100:>5.1f}%")
        rows.append({"W":W,"ctx_ms":ms,"causal":{"acc":ca,"macro_f1":cf,"hard":ch},
                     "centered":{"acc":na,"macro_f1":nf,"hard":nh}})

    best_c = max(rows, key=lambda r: r["causal"]["acc"])
    best_n = max(rows, key=lambda r: r["centered"]["acc"])
    print("\n" + "═"*76)
    print("[v52 판정 — 온라인 평활로 회수한 정직한 성능]")
    print(f"  기준 프레임(W=1)         : acc {base_acc*100:.2f}%  macroF1 {base_f1:.4f}")
    print(f"  인과적 최고 (W={best_c['W']}, {best_c['ctx_ms']}ms) : acc {best_c['causal']['acc']*100:.2f}%  "
          f"macroF1 {best_c['causal']['macro_f1']:.4f}  hard {best_c['causal']['hard']*100:.1f}%  "
          f"(+{(best_c['causal']['acc']-base_acc)*100:.2f}p)")
    print(f"  중심 최고 (W={best_n['W']}, {best_n['ctx_ms']}ms)   : acc {best_n['centered']['acc']*100:.2f}%  "
          f"macroF1 {best_n['centered']['macro_f1']:.4f}  hard {best_n['centered']['hard']*100:.1f}%  "
          f"(+{(best_n['centered']['acc']-base_acc)*100:.2f}p)")
    print(f"  oracle 반복 다수결(상한) : acc 88.04%")
    recov = (best_n['centered']['acc']-base_acc)/(0.8804-base_acc)*100
    print(f"  → 중심 평활이 프레임→oracle 격차의 {recov:.0f}%를 '분할 없이' 회수")

    json.dump({"model":"v52_temporal_smoothing",
               "baseline_window":{"acc":base_acc,"macro_f1":base_f1,"hard":base_hard},
               "oracle_vote_ref":{"acc":0.8804,"macro_f1":0.8457,"hard":0.8388},
               "sweep":rows,"best_causal":best_c,"best_centered":best_n},
              open(f"{OUT_DIR}/result_v52.json","w"),indent=2)
    print(f"\n저장: {OUT_DIR}/result_v52.json")


if __name__=="__main__":
    main()
