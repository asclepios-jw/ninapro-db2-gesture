"""
v57 (안정화 1/3): 통계적 안정성 — champion 앙상블의 신뢰구간(CI) + seed 분산.

목적: "숫자가 운이 아니라 안정적"임을 증명(MiniMax의 '+0.1p는 노이즈' 지적 대응).
방법(재학습 없음, 추론만):
  1. champion = 8모델 앙상블 확률 평균 → test 프레임/동작단위 지표(점추정).
  2. 부트스트랩 CI: 피험자 40명을 복원추출(B=2000) → 각 표본에서 지표 재계산 →
     95% percentile CI. 피험자 간 분산(일반화 불확실성)을 반영.
  3. seed 분산: 재현 멤버(v44 rep1/2, v45 rep2)를 교체한 앙상블들의 프레임 spread.
출력: results/v57/stability_v57.json
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention
from models_v41 import SubjectAwareResCNNLSTMAttention
from models_v44 import SubjectAwareCNNLSTMAttentionFeat
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

D="/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
R="/mnt/data/medical_project/ninapro_db2_project/results"
OUT=f"{R}/v57"; os.makedirs(OUT, exist_ok=True)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD={28,31,4,26,7}; NCL=50; B=2000

CHAMP=[(f"{R}/v44/best_model_v44.pth","v44"),(f"{R}/v44_rep1/best_model_v44.pth","v44"),
       (f"{R}/v44_rep2/best_model_v44.pth","v44"),(f"{R}/v45/best_model_v45.pth","v45"),
       (f"{R}/v45_rep2/best_model_v45.pth","v45"),(f"{R}/v40/best_model_v40.pth","v8"),
       (f"{R}/v41/best_model_v41.pth","v41"),(f"{R}/v38/best_model_v38.pth","v8")]

class DS(Dataset):
    def __init__(s,X,y,sub): s.X=X; s.y=y.astype(np.int64); s.s=sub.astype(np.int64)
    def __len__(s): return len(s.y)
    def __getitem__(s,i):
        x=np.ascontiguousarray(np.asarray(s.X[i]).transpose(1,0),dtype=np.float32)
        return torch.from_numpy(x), int(s.y[i]), int(s.s[i])

def build(k,nc,ncl,ns):
    if k=="v8":  return SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v41": return SubjectAwareResCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v44": return SubjectAwareCNNLSTMAttentionFeat(n_channels=nc,n_classes=ncl,n_subjects=ns)
    return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc,n_classes=ncl,n_subjects=ns)

@torch.no_grad()
def member_probs(pth,k,loader,nc,ncl,ns):
    m=build(k,nc,ncl,ns).to(DEV); m.load_state_dict(torch.load(pth,map_location=DEV,weights_only=True)); m.eval()
    out=[F.softmax(m(xb.to(DEV),sb.to(DEV)),1).cpu().numpy() for xb,_,sb in loader]
    del m; torch.cuda.empty_cache()
    return np.concatenate(out,0)

def pm_reduce(P,y,subs):
    """동작단위: 연속 같은 (subject,label) 구간 다수결. 반환: inst 예측/라벨/subject."""
    pred=P.argmax(1)
    key=subs.astype(np.int64)*1000+y.astype(np.int64)
    ch=np.concatenate([[True],key[1:]!=key[:-1]]); ids=np.cumsum(ch)-1
    iy,ip,isb=[],[],[]
    for i in range(ids.max()+1):
        m=ids==i; iy.append(y[m][0]); isb.append(subs[m][0]); ip.append(np.bincount(pred[m],minlength=NCL).argmax())
    return np.array(ip),np.array(iy),np.array(isb)

def metrics(pred,y,subs):
    h=np.isin(subs,list(HARD))
    return dict(acc=float(accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,average="macro",zero_division=0,labels=range(NCL))),
                hard=float(accuracy_score(y[h],pred[h])), easy=float(accuracy_score(y[~h],pred[~h])))

def boot_ci(pred,y,subs):
    """피험자 복원추출 부트스트랩 → acc/hard의 95% CI."""
    uniq=np.unique(subs); accs=[]; hards=[]
    corr=(pred==y).astype(np.float64)
    for _ in range(B):
        samp=np.random.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([np.where(subs==s)[0] for s in samp])
        accs.append(corr[idx].mean())
        hs=[s for s in samp if s in HARD]
        if hs:
            hidx=np.concatenate([np.where(subs==s)[0] for s in hs]); hards.append(corr[hidx].mean())
    def ci(a): a=np.array(a); return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5)),float(a.std())]
    return ci(accs), ci(hards)

def main():
    t0=time.time()
    Xte=np.load(f"{D}/X_test.npy",mmap_mode="r"); yte=np.load(f"{D}/y_test.npy"); ste=np.load(f"{D}/s_test.npy")
    ytr=np.load(f"{D}/y_train.npy")
    le=LabelEncoder(); le.fit(ytr); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{D}/s_train.npy").max())
    loader=DataLoader(DS(Xte,yte,ste),batch_size=256,shuffle=False,num_workers=3,pin_memory=True)

    print(f"멤버 {len(CHAMP)}개 확률 계산...")
    Ps=[member_probs(p,k,loader,nc,ncl,ns) for p,k in CHAMP]
    Pens=np.mean(Ps,axis=0)

    # 점추정
    fpred=Pens.argmax(1); fm=metrics(fpred,yte,ste)
    ip,iy,isb=pm_reduce(Pens,yte,ste); pm=metrics(ip,iy,isb)
    print(f"\n[champion 점추정]")
    print(f"  프레임      acc {fm['acc']*100:.2f}%  macroF1 {fm['macro_f1']:.4f}  hard {fm['hard']*100:.2f}%")
    print(f"  동작단위    acc {pm['acc']*100:.2f}%  macroF1 {pm['macro_f1']:.4f}  hard {pm['hard']*100:.2f}%")

    # 부트스트랩 CI
    print(f"\n부트스트랩 CI (B={B}, 피험자 복원추출)...")
    (fa_lo,fa_hi,fa_sd),(fh_lo,fh_hi,fh_sd)=boot_ci(fpred,yte,ste)
    (pa_lo,pa_hi,pa_sd),(ph_lo,ph_hi,ph_sd)=boot_ci(ip,iy,isb)
    print(f"  프레임   acc  {fm['acc']*100:.2f}%  95%CI [{fa_lo*100:.2f}, {fa_hi*100:.2f}]  (±{fa_sd*100:.2f})")
    print(f"  프레임   hard {fm['hard']*100:.2f}%  95%CI [{fh_lo*100:.2f}, {fh_hi*100:.2f}]")
    print(f"  동작단위 acc  {pm['acc']*100:.2f}%  95%CI [{pa_lo*100:.2f}, {pa_hi*100:.2f}]  (±{pa_sd*100:.2f})")
    print(f"  동작단위 hard {pm['hard']*100:.2f}%  95%CI [{ph_lo*100:.2f}, {ph_hi*100:.2f}]")

    # seed 분산: v44/v45 재현 멤버 교체 조합들의 프레임 acc spread
    print(f"\nseed 분산 (재현멤버 교체 조합의 프레임 acc)...")
    idx={p:i for i,(p,k) in enumerate(CHAMP)}
    base_others=[idx[f"{R}/v40/best_model_v40.pth"],idx[f"{R}/v41/best_model_v41.pth"],idx[f"{R}/v38/best_model_v38.pth"]]
    v44s=[idx[f"{R}/v44/best_model_v44.pth"],idx[f"{R}/v44_rep1/best_model_v44.pth"],idx[f"{R}/v44_rep2/best_model_v44.pth"]]
    v45s=[idx[f"{R}/v45/best_model_v45.pth"],idx[f"{R}/v45_rep2/best_model_v45.pth"]]
    seed_accs=[]
    for a in v44s:
        for b in v45s:
            sub=np.mean([Ps[a],Ps[b]]+[Ps[i] for i in base_others],axis=0)
            seed_accs.append(float(accuracy_score(yte,sub.argmax(1))))
    seed_accs=np.array(seed_accs)
    print(f"  조합 {len(seed_accs)}개: {seed_accs.min()*100:.2f}~{seed_accs.max()*100:.2f}%  "
          f"평균 {seed_accs.mean()*100:.2f}% ±{seed_accs.std()*100:.3f}p")

    json.dump({"model":"v57_stability","champion_members":len(CHAMP),"n_boot":B,
        "frame":{**fm,"acc_ci95":[fa_lo,fa_hi],"acc_sd":fa_sd,"hard_ci95":[fh_lo,fh_hi]},
        "per_movement":{**pm,"acc_ci95":[pa_lo,pa_hi],"acc_sd":pa_sd,"hard_ci95":[ph_lo,ph_hi]},
        "seed_variance_frame":{"n":len(seed_accs),"min":float(seed_accs.min()),"max":float(seed_accs.max()),
                               "mean":float(seed_accs.mean()),"std":float(seed_accs.std())},
        "elapsed_sec":time.time()-t0}, open(f"{OUT}/stability_v57.json","w"), indent=2)
    print(f"\n총 {time.time()-t0:.0f}s  저장: {OUT}/stability_v57.json")

if __name__=="__main__": main()
