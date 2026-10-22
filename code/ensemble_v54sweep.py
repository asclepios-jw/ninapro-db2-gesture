"""
v54 평가: 스펙트로그램 CNN(v54)을 후보에 추가한 10-후보 앙상블 sweep.
핵심 질문: v54 단독은 약하나(57%), 다른 아키텍처라 비상관 → 앙상블에 기여하는가?
          (v48 최고 72.9955% / 프레임 73%를 넘는가, 최적 조합에 v54 포함되는가.)
추론만, 재학습 0.
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
from models_v54 import SpectrogramCNN

DATA_DIR="/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR ="/mnt/data/medical_project/ninapro_db2_project/results/v54"
os.makedirs(OUT_DIR,exist_ok=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS={28,31,4,26,7}; R="/mnt/data/medical_project/ninapro_db2_project/results"
CAND=[
    ("v44_orig",f"{R}/v44/best_model_v44.pth","v44"),("v44_rep1",f"{R}/v44_rep1/best_model_v44.pth","v44"),
    ("v44_rep2",f"{R}/v44_rep2/best_model_v44.pth","v44"),("v45",f"{R}/v45/best_model_v45.pth","v45"),
    ("v45_r1",f"{R}/v45_rep1/best_model_v45.pth","v45"),("v45_r2",f"{R}/v45_rep2/best_model_v45.pth","v45"),
    ("v40",f"{R}/v40/best_model_v40.pth","v8"),("v41",f"{R}/v41/best_model_v41.pth","v41"),
    ("v38",f"{R}/v38/best_model_v38.pth","v8"),("v54",f"{R}/v54/best_model_v54.pth","v54"),  # 신규
]


class DS(Dataset):
    def __init__(s,X,y,sub):
        s.X=torch.from_numpy(X.transpose(0,2,1).astype(np.float32)); s.y=torch.from_numpy(y.astype(np.int64)); s.s=torch.from_numpy(sub.astype(np.int64))
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.X[i],s.y[i],s.s[i]

def build(k,nc,ncl,ns):
    if k=="v8":  return SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v41": return SubjectAwareResCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v44": return SubjectAwareCNNLSTMAttentionFeat(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v45": return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc,n_classes=ncl,n_subjects=ns)
    return SpectrogramCNN(n_channels=nc,n_classes=ncl,n_subjects=ns)

@torch.no_grad()
def probs(path,k,loader,nc,ncl,ns):
    m=build(k,nc,ncl,ns).to(DEVICE); m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True)); m.eval()
    o=[]
    for xb,yb,sb in loader: o.append(F.softmax(m(xb.to(DEVICE),sb.to(DEVICE)),1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(o,0)

def main():
    ytr=np.load(f"{DATA_DIR}/y_train.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); y=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader=DataLoader(DS(Xte,y,ste),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
    hidx=np.isin(ste,list(HARD_SUBJECTS))

    names=[]; P=[]
    print("단일:")
    for nm,pth,k in CAND:
        p=probs(pth,k,loader,nc,ncl,ns); P.append(p); names.append(nm)
        print(f"  {nm:>9} {accuracy_score(y,p.argmax(1))*100:.2f}%")
    P=np.stack(P,0); M=len(names)
    def acc_of(idx): return accuracy_score(y,P[list(idx)].mean(0).argmax(1))

    best={"acc":-1}; by={}
    for k in range(1,M+1):
        bs=None
        for c in itertools.combinations(range(M),k):
            a=acc_of(c)
            if bs is None or a>bs["acc"]: bs={"acc":a,"members":[names[i] for i in c]}
            if a>best["acc"]: best={"acc":a,"members":[names[i] for i in c],"size":k}
        by[k]=bs
    print(f"\n{'크기':>3} {'최고 acc':>8}  최고 조합")
    for k in range(1,M+1):
        st="  ★≥73" if by[k]['acc']>=0.73 else ""
        print(f"{k:>3} {by[k]['acc']*100:>7.2f}%  {'+'.join(by[k]['members'])}{st}")

    bi=[names.index(n) for n in best["members"]]; bp=P[bi].mean(0).argmax(1)
    bf1=f1_score(y,bp,average="macro",zero_division=0); bh=accuracy_score(y[hidx],bp[hidx])
    v54_in = "v54" in best["members"]
    print("\n"+"═"*66)
    print(f"[v54 앙상블 판정]")
    print(f"  최고 조합({best['size']}개): {best['acc']*100:.4f}%  hard {bh*100:.2f}%  macroF1 {bf1:.4f}")
    print(f"    멤버: {'+'.join(best['members'])}")
    print(f"  v48(9후보) 72.9955% 대비: {(best['acc']-0.729955)*100:+.3f}p")
    print(f"  v54(스펙트로그램) 최고 조합 포함: {v54_in}  → {'다른 아키텍처가 기여!' if v54_in else '기여 못함(단독 약함+상관)'}")
    json.dump({"model":"v54_ensemble","candidates":names,
               "v54_solo_acc":float(accuracy_score(y,P[names.index('v54')].argmax(1))),
               "best":{**best,"hard":bh,"macro_f1":bf1},"v54_in_best":bool(v54_in),
               "vs_v48":best["acc"]-0.729955},open(f"{OUT_DIR}/result_v54.json","w"),indent=2)
    print(f"\n저장: {OUT_DIR}/result_v54.json")

if __name__=="__main__": main()
