"""
v64: 강건 앙상블 구성·검증 — 채널드롭 3멤버(v61+v62+v63) vs 대조군.

비교 3종(모두 앙상블):
  robust3   = 채널드롭 {v61(v8), v62(v44), v63(v45)}          ← 강건 배포 후보
  orig3     = 비채널드롭 {v40(v8), v44, v45} (같은 아키텍처)   ← 채널드롭 효과 격리
  champ8    = 기존 champion 8멤버 (참조)
각각 clean(프레임/동작단위/hard) + 전극 고장(1/2/3개) 곡선.
출력: results/v64/robust_ensemble_v64.json
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from stability_v57 import build, DS, pm_reduce, metrics, CHAMP, D, R, HARD

OUT=f"{R}/v64"; os.makedirs(OUT, exist_ok=True)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROBUST3=[(f"{R}/v61/best_model_v61.pth","v8"),(f"{R}/v62/best_model.pth","v44"),(f"{R}/v63/best_model.pth","v45")]
ORIG3=[(f"{R}/v40/best_model_v40.pth","v8"),(f"{R}/v44/best_model_v44.pth","v44"),(f"{R}/v45/best_model_v45.pth","v45")]

def load(members,nc,ncl,ns):
    ms=[]
    for p,k in members:
        m=build(k,nc,ncl,ns).to(DEV); m.load_state_dict(torch.load(p,map_location=DEV,weights_only=True)); m.eval(); ms.append(m)
    return ms

@torch.no_grad()
def ens_probs(models, loader, elecs=None):
    ch=np.concatenate([np.array(elecs),np.array(elecs)+12]) if elecs else None
    outs=[]; ys=[]; ss=[]
    for xb,yb,sb in loader:
        xb=xb.to(DEV)
        if ch is not None: xb[:,ch,:]=0.0
        sb=sb.to(DEV); prob=None
        for m in models:
            p=F.softmax(m(xb,sb),1); prob=p if prob is None else prob+p
        outs.append((prob/len(models)).cpu().numpy()); ys.append(yb.numpy()); ss.append(sb.cpu().numpy())
    return np.concatenate(outs),np.concatenate(ys),np.concatenate(ss)

def drop_acc(models, loader, k):
    accs=[]
    for t in range(3):
        elec=list(np.random.RandomState(100+t).choice(12,size=k,replace=False))
        P,y,s=ens_probs(models,loader,elec); accs.append(accuracy_score(y,P.argmax(1)))
    return float(np.mean(accs))

def main():
    t0=time.time()
    Xte=np.load(f"{D}/X_test.npy",mmap_mode="r"); yte=np.load(f"{D}/y_test.npy"); ste=np.load(f"{D}/s_test.npy")
    ytr=np.load(f"{D}/y_train.npy"); le=LabelEncoder(); le.fit(ytr); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{D}/s_train.npy").max())
    loader=DataLoader(DS(Xte,yte,ste),batch_size=256,shuffle=False,num_workers=3,pin_memory=True)

    res={}
    for name,mem in [("robust3(chdrop)",ROBUST3),("orig3",ORIG3),("champ8",CHAMP)]:
        ms=load(mem,nc,ncl,ns)
        P,y,s=ens_probs(ms,loader)
        fm=metrics(P.argmax(1),y,s); ip,iy,isb=pm_reduce(P,y,s); pm=metrics(ip,iy,isb)
        curve={"clean":fm["acc"],"drop1":drop_acc(ms,loader,1),"drop2":drop_acc(ms,loader,2),"drop3":drop_acc(ms,loader,3)}
        res[name]={"frame":fm["acc"],"per_movement":pm["acc"],"hard":fm["hard"],"drop":curve}
        print(f"[{name}] clean 프레임 {fm['acc']*100:.2f}% 동작단위 {pm['acc']*100:.2f}% hard {fm['hard']*100:.2f}%")
        del ms; torch.cuda.empty_cache()

    print("\n[전극 고장 강건성] 프레임 acc")
    print(f"  {'조건':<7}{'champ8':>9}{'orig3':>9}{'robust3':>10}")
    for kkey in ["clean","drop1","drop2","drop3"]:
        print(f"  {kkey:<7}{res['champ8']['drop'][kkey]*100:>8.2f}%{res['orig3']['drop'][kkey]*100:>8.2f}%{res['robust3(chdrop)']['drop'][kkey]*100:>9.2f}%")

    json.dump({"model":"v64_robust_ensemble","results":res,"elapsed_sec":time.time()-t0},
              open(f"{OUT}/robust_ensemble_v64.json","w"),indent=2)
    print(f"\n총 {time.time()-t0:.0f}s  저장: {OUT}/robust_ensemble_v64.json")

if __name__=="__main__": main()
