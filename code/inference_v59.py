"""
v59 (안정화 3/3): 재현성 + 엔드투엔드 추론 파이프라인.

목적: "남이 그대로 재현·구동"할 수 있는 단일 추론 진입점을 고정한다.
  · 시드 고정(재현성).
  · CONFIG로 배포 구성 선택(champion 8 / 권장 diverse 5 / 단일 v45).
  · predict(X, subs) → 앙상블 확률 → 프레임 예측 + (선택)인과 평활 + 동작단위.
  · main(): test에서 재현 검증 — 보고 수치(프레임 73.0 / 동작단위 88.0)와 대조.
재학습 0, 추론만. 결과: results/v59/inference_v59.json
"""
import os, sys, json, time, random
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from stability_v57 import DS, member_probs, pm_reduce, metrics, CHAMP, D, R, HARD, NCL

OUT=f"{R}/v59"; os.makedirs(OUT, exist_ok=True)

# ── 재현성: 시드 고정 ────────────────────────────────────────────────────────
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# ── 배포 구성 선택 ───────────────────────────────────────────────────────────
CONFIGS={
    "champion_8": [p for p,_ in CHAMP],
    "recommended_5": [f"{R}/v44/best_model_v44.pth",f"{R}/v45/best_model_v45.pth",
                      f"{R}/v40/best_model_v40.pth",f"{R}/v41/best_model_v41.pth",f"{R}/v38/best_model_v38.pth"],
    "single_v45": [f"{R}/v45/best_model_v45.pth"],
}
KIND={f"{R}/v44/best_model_v44.pth":"v44",f"{R}/v44_rep1/best_model_v44.pth":"v44",
      f"{R}/v44_rep2/best_model_v44.pth":"v44",f"{R}/v45/best_model_v45.pth":"v45",
      f"{R}/v45_rep2/best_model_v45.pth":"v45",f"{R}/v40/best_model_v40.pth":"v8",
      f"{R}/v41/best_model_v41.pth":"v41",f"{R}/v38/best_model_v38.pth":"v8"}

def causal_smooth(P, subs, W=5):
    """피험자 시간순 스트림에서 확률의 인과(과거만) 이동평균. 실시간용."""
    out=np.empty_like(P)
    for s in np.unique(subs):
        idx=np.where(subs==s)[0]           # 저장순=시간순
        q=P[idx]; c=np.cumsum(q,axis=0)
        sm=q.copy()
        for t in range(len(q)):
            a=max(0,t-W+1); sm[t]=(c[t]-(c[a-1] if a>0 else 0))/(t-a+1)
        out[idx]=sm
    return out

def ensemble_probs(members, loader, nc, ncl, ns):
    """배포 추론 핵심: 선택된 멤버 확률 평균."""
    Ps=[member_probs(p, KIND[p], loader, nc, ncl, ns) for p in members]
    return np.mean(Ps, axis=0)

def main():
    t0=time.time()
    Xte=np.load(f"{D}/X_test.npy",mmap_mode="r"); yte=np.load(f"{D}/y_test.npy"); ste=np.load(f"{D}/s_test.npy")
    ytr=np.load(f"{D}/y_train.npy")
    le=LabelEncoder(); le.fit(ytr); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{D}/s_train.npy").max())
    loader=DataLoader(DS(Xte,yte,ste),batch_size=256,shuffle=False,num_workers=3,pin_memory=True)

    EXPECT={"champion_8":(0.7300,0.8804),"recommended_5":(0.725,0.875),"single_v45":(0.7031,0.8594)}
    res={}
    for name,members in CONFIGS.items():
        P=ensemble_probs(members, loader, nc, ncl, ns)
        fm=metrics(P.argmax(1),yte,ste)
        ip,iy,isb=pm_reduce(P,yte,ste); pm=metrics(ip,iy,isb)
        Psm=causal_smooth(P,ste,W=5); fsm=metrics(Psm.argmax(1),yte,ste)
        res[name]={"frame":fm,"frame_causal_smooth":fsm,"per_movement":pm,"n_members":len(members)}
        ef,ep=EXPECT[name]; ok="✓" if abs(fm["acc"]-ef)<0.005 and abs(pm["acc"]-ep)<0.01 else "✗"
        print(f"[{name}] ({len(members)}모델)")
        print(f"  프레임 {fm['acc']*100:.2f}%  인과평활 {fsm['acc']*100:.2f}%  동작단위 {pm['acc']*100:.2f}%  hard {fm['hard']*100:.2f}%")
        print(f"  재현검증(기대 프레임{ef*100:.1f}/동작{ep*100:.1f}): {ok}")

    json.dump({"model":"v59_inference_pipeline","seed":SEED,"configs":res,"elapsed_sec":time.time()-t0},
              open(f"{OUT}/inference_v59.json","w"), indent=2)
    print(f"\n총 {time.time()-t0:.0f}s  저장: {OUT}/inference_v59.json")
    print("→ 배포 추론 API: ensemble_probs(CONFIGS[구성], loader, ...) [+ causal_smooth 선택]")

if __name__=="__main__": main()
