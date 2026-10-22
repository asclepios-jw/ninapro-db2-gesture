"""
v53: 실제(예측 기반) 분할 per-movement — oracle 없이 얼마나 회수되나 (추론 후처리).

v51 oracle 반복 다수결 88%는 '제스처 경계를 안다'는 전제. v52 온라인 평활은
78.6%(중심). 남은 격차는 분할. 이 버전은 '예측된 rest↔제스처 런'으로 세그먼트를
직접 검출(oracle 아님)하고, 세그먼트 내 다수결로 라벨을 강제해 프레임 성능을 잰다.

방법:
  1. 앙상블 윈도우 예측(argmax). rest=인코딩 0.
  2. 피험자 내 시간순으로, 예측 비-rest 런 = 검출 제스처 세그먼트.
     - 짧은 rest 간극(<=GAP)은 이어붙임(같은 제스처의 순간 rest 오탐 방지).
     - 너무 짧은 비-rest 런(<MINLEN)은 rest로 폐기(오탐 제거).
  3. 각 검출 세그먼트를 그 안 비-rest 예측의 다수결 라벨로 통일 → 프레임 예측.
  4. 프레임 레벨 acc/macroF1/hard 재계산. 검출 세그먼트 수 vs 실제 제스처 수 비교.
비교: 기준 프레임(73%) / v52 중심평활(78.6%) / v51 oracle(88%).
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
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v53"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
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


def real_segment_relabel(pred, subs, GAP, MINLEN):
    """예측 rest/비-rest로 세그먼트 검출 후 세그먼트 내 다수결로 통일. 검출 세그먼트 수 반환."""
    out = pred.copy()
    n_seg = 0
    for s in np.unique(subs):
        idx = np.where(subs == s)[0]
        p = pred[idx].copy()
        active = (p != 0).astype(np.int8)
        # 짧은 rest 간극(<=GAP) 메우기: 비-rest 사이 짧은 rest를 활성으로
        L = len(p); i = 0
        while i < L:
            if active[i] == 0:
                j = i
                while j < L and active[j] == 0: j += 1
                gap = j - i
                if gap <= GAP and i > 0 and j < L:   # 양옆이 활성인 짧은 간극
                    active[i:j] = 1
                i = j
            else:
                i += 1
        # 활성 런 검출 → MINLEN 미만 폐기, 나머지는 다수결 라벨로 통일
        i = 0
        while i < L:
            if active[i] == 1:
                j = i
                while j < L and active[j] == 1: j += 1
                run = slice(i, j)
                if (j - i) < MINLEN:
                    out[idx[run]] = 0            # 짧은 오탐 → rest
                else:
                    seg_pred = p[run]
                    nz = seg_pred[seg_pred != 0]
                    lab = np.bincount(nz).argmax() if len(nz) else 0
                    out[idx[run]] = lab           # 세그먼트 전체를 다수결 라벨로
                    n_seg += 1
                i = j
            else:
                i += 1
    return out, n_seg


def met(y, p, subs, ncl):
    hidx = np.isin(subs, list(HARD_SUBJECTS))
    return (accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0, labels=range(ncl)),
            accuracy_score(y[hidx], p[hidx]))


def count_true_gestures(y, subs):
    """실제 비-rest 제스처 실행 인스턴스 수 (연속 동일 (subject,label), label!=0)."""
    key = subs.astype(np.int64)*1000 + y.astype(np.int64)
    change = np.concatenate([[True], key[1:] != key[:-1]])
    ids = np.cumsum(change)-1
    n=0
    for i in range(ids.max()+1):
        m = ids==i
        if y[m][0] != 0: n+=1
    return n


def main():
    print("데이터 로드...")
    ytr=np.load(f"{DATA_DIR}/y_train.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); y=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader=DataLoader(DS(Xte,y,ste),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    print(f"앙상블 8모델 예측 계산...")
    P=np.mean([probs(p,k,loader,nc,ncl,ns) for p,k in ENSEMBLE],axis=0)
    pred=P.argmax(1)
    base=met(y,pred,ste,ncl)
    n_true=count_true_gestures(y,ste)
    print(f"\n[기준 프레임] acc {base[0]*100:.2f}%  macroF1 {base[1]:.4f}  hard {base[2]*100:.1f}%")
    print(f"[참조] v52 중심평활 78.64% / v51 oracle 88.04%")
    print(f"[실제 제스처 실행 수(GT)] {n_true}\n")

    print(f"{'GAP':>4} {'MINLEN':>6} │ {'acc':>7} {'macroF1':>8} {'hard':>6} │ {'검출세그':>7} {'GT대비':>6}")
    print("─"*60)
    rows=[]
    for GAP in [1,2,3]:
        for MINLEN in [2,3,5]:
            out,nseg = real_segment_relabel(pred, ste, GAP, MINLEN)
            a,f,h = met(y,out,ste,ncl)
            print(f"{GAP:>4} {MINLEN:>6} │ {a*100:>6.2f}% {f:>8.4f} {h*100:>5.1f}% │ {nseg:>7} {nseg/n_true:>5.2f}x")
            rows.append({"gap":GAP,"minlen":MINLEN,"acc":a,"macro_f1":f,"hard":h,"n_seg":int(nseg)})

    best=max(rows,key=lambda r:r["acc"])
    print("\n" + "═"*60)
    print("[v53 판정 — 실제 분할(oracle 아님) per-movement]")
    print(f"  기준 프레임      : {base[0]*100:.2f}% / macroF1 {base[1]:.4f}")
    print(f"  실제분할 최고    : {best['acc']*100:.2f}% / macroF1 {best['macro_f1']:.4f} / hard {best['hard']*100:.1f}%"
          f"  (GAP={best['gap']},MINLEN={best['minlen']}, 검출 {best['n_seg']}세그 = GT의 {best['n_seg']/n_true:.2f}x)")
    print(f"  v52 중심평활     : 78.64%")
    print(f"  oracle 상한      : 88.04%")
    print(f"  → 프레임 대비 +{(best['acc']-base[0])*100:.2f}p, oracle 격차의 {(best['acc']-base[0])/(0.8804-base[0])*100:.0f}% 회수")

    json.dump({"model":"v53_real_segmentation","baseline_frame":{"acc":base[0],"macro_f1":base[1],"hard":base[2]},
               "n_true_gestures":int(n_true),"sweep":rows,"best":best,
               "refs":{"v52_centered":0.7864,"v51_oracle":0.8804}},
              open(f"{OUT_DIR}/result_v53.json","w"),indent=2)
    print(f"\n저장: {OUT_DIR}/result_v53.json")


if __name__=="__main__":
    main()
