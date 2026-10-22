"""
v60 (안정화 4/4, 밀렸던 S4): 강건성 테스트 — champion이 실사용 교란에 완만히
                              저하되는지 검증.

교란(추론 시 test 입력에 적용, 재학습 0):
  1. 가우시안 잡음: σ(신호 std 대비) ∈ {0.05,0.1,0.2,0.4}
  2. 전극 고장(채널 손실): k개 전극 0으로 → raw i + envelope i+12 쌍 제거.
     k ∈ {1,2,3}, 각 3회 무작위 평균.
  3. 게인 드리프트: 전체 배수 g ∈ {0.5,0.7,1.5,2.0}
champion 8모델을 메모리에 상주시키고, test 4만 윈도우 부분표본으로 프레임 acc/hard.
출력: results/v60/robustness_v60.json
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from stability_v57 import build, CHAMP, D, R, HARD

OUT=f"{R}/v60"; os.makedirs(OUT, exist_ok=True)
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)
NSUB=40000; BS=256

def load_all(nc,ncl,ns):
    ms=[]
    for p,k in CHAMP:
        m=build(k,nc,ncl,ns).to(DEV); m.load_state_dict(torch.load(p,map_location=DEV,weights_only=True)); m.eval()
        ms.append(m)
    return ms

@torch.no_grad()
def eval_pert(models, X, y, subs, sd, pert):
    """pert: (xb_tensor)->xb_tensor. 프레임 acc/hard 반환."""
    preds=[]
    for a in range(0,len(y),BS):
        xb=np.ascontiguousarray(np.asarray(X[a:a+BS]).transpose(0,2,1),dtype=np.float32)
        xb=torch.from_numpy(xb).to(DEV)
        if pert is not None: xb=pert(xb)
        sb=torch.from_numpy(subs[a:a+BS].astype(np.int64)).to(DEV)
        prob=None
        for m in models:
            p=F.softmax(m(xb,sb),1); prob=p if prob is None else prob+p
        preds.append(prob.argmax(1).cpu().numpy())
    pred=np.concatenate(preds); h=np.isin(subs,list(HARD))
    return float(accuracy_score(y,pred)), float(accuracy_score(y[h],pred[h]))

def main():
    t0=time.time()
    Xte=np.load(f"{D}/X_test.npy",mmap_mode="r"); yte=np.load(f"{D}/y_test.npy"); ste=np.load(f"{D}/s_test.npy")
    ytr=np.load(f"{D}/y_train.npy"); le=LabelEncoder(); le.fit(ytr); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{D}/s_train.npy").max())
    # 부분표본(재현 고정) — 정렬해 mmap 순차접근
    idx=np.sort(np.random.choice(len(yte),size=min(NSUB,len(yte)),replace=False))
    Xs=np.asarray(Xte[idx]); ys=yte[idx]; ss=ste[idx]
    print(f"부분표본 {len(ys):,} 윈도우, 멤버 {len(CHAMP)}개 상주")
    models=load_all(nc,ncl,ns)

    # 기준(교란 없음)
    base_acc,base_hard=eval_pert(models,Xs,ys,ss,None,None)
    print(f"\n[기준] 프레임 {base_acc*100:.2f}%  hard {base_hard*100:.2f}%  (부분표본)")
    res={"baseline":{"acc":base_acc,"hard":base_hard},"noise":{},"electrode_drop":{},"gain":{}}

    print("\n[잡음]  σ    프레임    hard    저하")
    sig=float(np.std(Xs))
    for s in [0.05,0.1,0.2,0.4]:
        f=lambda xb,s=s: xb + torch.randn_like(xb)*(s*sig)
        a,h=eval_pert(models,Xs,ys,ss,None,f)
        res["noise"][s]={"acc":a,"hard":h}
        print(f"       {s:<5} {a*100:.2f}%  {h*100:.2f}%  {(a-base_acc)*100:+.2f}p")

    print("\n[전극고장]  k    프레임    hard    저하  (3회 무작위 평균)")
    for k in [1,2,3]:
        accs=[];hards=[]
        for t in range(3):
            elec=np.random.RandomState(100+t).choice(12,size=k,replace=False)
            ch=np.concatenate([elec,elec+12])
            def f(xb,ch=ch):
                xb=xb.clone(); xb[:,ch,:]=0.0; return xb
            a,h=eval_pert(models,Xs,ys,ss,None,f); accs.append(a);hards.append(h)
        a=float(np.mean(accs));h=float(np.mean(hards))
        res["electrode_drop"][k]={"acc":a,"hard":h}
        print(f"          {k}    {a*100:.2f}%  {h*100:.2f}%  {(a-base_acc)*100:+.2f}p")

    print("\n[게인]  g     프레임    hard    저하")
    for g in [0.5,0.7,1.5,2.0]:
        f=lambda xb,g=g: xb*g
        a,h=eval_pert(models,Xs,ys,ss,None,f)
        res["gain"][g]={"acc":a,"hard":h}
        print(f"       {g:<5} {a*100:.2f}%  {h*100:.2f}%  {(a-base_acc)*100:+.2f}p")

    json.dump({"model":"v60_robustness","n_sub":len(ys),"baseline":res["baseline"],
               "noise":res["noise"],"electrode_drop":res["electrode_drop"],"gain":res["gain"],
               "elapsed_sec":time.time()-t0}, open(f"{OUT}/robustness_v60.json","w"), indent=2)
    print(f"\n총 {time.time()-t0:.0f}s  저장: {OUT}/robustness_v60.json")

if __name__=="__main__": main()
