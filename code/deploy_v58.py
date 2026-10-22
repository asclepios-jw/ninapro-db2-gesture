"""
v58 (안정화 2/3): 배포 구성 확정 — 효율곡선 + 조합 robustness + 단일 fallback.

목적: "최종 배포 모델"을 하나로 고정. 추론 비용(모델 수) 대비 성능을 정량화해
      권장 구성을 정한다. 재학습 0, 기존 체크포인트 재사용.
방법:
  1. 8멤버 확률을 test에서 1회 계산(stability_v57 재사용).
  2. 효율곡선: 모든 비어있지 않은 부분집합(255개)을 크기별로 묶어 프레임 acc의
     평균/최소/최대 → "N개면 얼마" + 조합 robustness(같은 크기 spread).
  3. 단일 fallback: 개별 멤버 최고 성능(프레임+동작단위).
  4. 권장: 풀8 대비 0.3p 이내로 근접하는 최소 N.
출력: results/v58/deploy_v58.json
"""
import os, sys, json, time, itertools
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as _np

sys.path.insert(0, os.path.dirname(__file__))
from stability_v57 import (DS, member_probs, pm_reduce, metrics, CHAMP, D, R, HARD, NCL)

OUT=f"{R}/v58"; os.makedirs(OUT, exist_ok=True)

def main():
    t0=time.time()
    Xte=np.load(f"{D}/X_test.npy",mmap_mode="r"); yte=np.load(f"{D}/y_test.npy"); ste=np.load(f"{D}/s_test.npy")
    ytr=np.load(f"{D}/y_train.npy")
    le=LabelEncoder(); le.fit(ytr); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{D}/s_train.npy").max())
    loader=DataLoader(DS(Xte,yte,ste),batch_size=256,shuffle=False,num_workers=3,pin_memory=True)

    names=[os.path.basename(os.path.dirname(p)) for p,_ in CHAMP]
    print(f"멤버 {len(CHAMP)}개 확률 계산: {names}")
    Ps=[member_probs(p,k,loader,nc,ncl,ns) for p,k in CHAMP]
    M=len(Ps)

    def frame_acc(subset):
        avg=np.mean([Ps[i] for i in subset],axis=0)
        return accuracy_score(yte, avg.argmax(1))

    # 1) 단일 fallback
    singles=[(i,frame_acc([i])) for i in range(M)]
    best_i,best_single=max(singles,key=lambda x:x[1])
    avg_best=Ps[best_i]
    fm_s=metrics(avg_best.argmax(1),yte,ste)
    ip,iy,isb=pm_reduce(avg_best,yte,ste); pm_s=metrics(ip,iy,isb)
    print(f"\n[단일 fallback] 최고 멤버 = {names[best_i]}")
    print(f"  프레임 {fm_s['acc']*100:.2f}%  동작단위 {pm_s['acc']*100:.2f}%  hard(프레임) {fm_s['hard']*100:.2f}%")

    # 2) 효율곡선(모든 부분집합, 크기별 통계)
    print(f"\n[효율곡선] 크기별 프레임 acc (모든 조합)")
    curve={}
    for n in range(1,M+1):
        accs=[frame_acc(list(c)) for c in itertools.combinations(range(M),n)]
        curve[n]={"n_comb":len(accs),"mean":float(np.mean(accs)),"min":float(np.min(accs)),"max":float(np.max(accs))}
        print(f"  N={n}: mean {np.mean(accs)*100:.2f}%  [{np.min(accs)*100:.2f}~{np.max(accs)*100:.2f}]  ({len(accs)}조합)")

    full=curve[M]["mean"]
    # 3) 권장 N: 풀8 대비 0.3p 이내 최소 N
    rec_n=next((n for n in range(1,M+1) if curve[n]["mean"]>=full-0.003), M)
    print(f"\n[권장] 풀{M} 프레임 {full*100:.2f}% 기준, 0.3p 이내 최소 N = {rec_n}")
    print(f"  → N={rec_n}: {curve[rec_n]['mean']*100:.2f}% (추론비용 {rec_n}/{M})")

    # 권장 구성 대표(상위 N개 단일성능) 전체 지표
    top=[i for i,_ in sorted(singles,key=lambda x:-x[1])[:rec_n]]
    avg_rec=np.mean([Ps[i] for i in top],axis=0)
    fm_r=metrics(avg_rec.argmax(1),yte,ste)
    ipr,iyr,isbr=pm_reduce(avg_rec,yte,ste); pm_r=metrics(ipr,iyr,isbr)
    print(f"  권장구성({[names[i] for i in top]}): 프레임 {fm_r['acc']*100:.2f}% 동작단위 {pm_r['acc']*100:.2f}% hard {fm_r['hard']*100:.2f}%")

    json.dump({"model":"v58_deploy_config","members":names,
        "single_fallback":{"member":names[best_i],"frame":fm_s,"per_movement":pm_s},
        "efficiency_curve":curve,"full_ensemble_frame":full,
        "recommended_n":rec_n,"recommended_members":[names[i] for i in top],
        "recommended":{"frame":fm_r,"per_movement":pm_r},
        "elapsed_sec":time.time()-t0}, open(f"{OUT}/deploy_v58.json","w"), indent=2)
    print(f"\n총 {time.time()-t0:.0f}s  저장: {OUT}/deploy_v58.json")

if __name__=="__main__": main()
