"""
v61 (개념검증): 채널 드롭아웃 증강으로 전극 고장 강건성 하드닝 (base 1개).

동기: v60에서 전극 1개 고장 = −13.5p(앙상블) 취약성 발견. 훈련 중 무작위로
      전극을 0으로 만드는 증강(channel dropout)을 넣으면, 모델이 특정 전극에
      과의존하지 않아 고장에 강건해질 것.

설계(깔끔한 A/B, base 1개):
  baseline = v40 (v8 아키텍처 + v40 레시피, 채널드롭 없음)  ← 기존 best_model_v40.pth
  v61      = v8 아키텍처 + v40 레시피 + 채널드롭 증강        ← 이 스크립트
  차이는 오직 채널드롭 증강. 검증:
    (1) clean 정확도(프레임/동작단위/hard) 보존되나?
    (2) 전극 고장(1/2/3개) 저하가 baseline보다 작아지나?

데이터: full np.load(RAM) + 샘플별 transpose(FrugalEMGDataset) → 빠르고 OOM 없음.
출력: results/v61/best_model_v61.pth, result_v61.json
"""
import os, sys, time, json
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention
from train_cnn_v40 import (focal_loss, aug_jitter, aug_magnitude_warp, aug_time_warp, mixup_batch,
                           N_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY, T_MAX, DEVICE, HARD_SUBJECTS,
                           JITTER_SIGMA, MAGWARP_SIGMA, TIMEWARP_SIGMA, WARP_KNOTS, MIXUP_ALPHA, FOCAL_GAMMA)
from train_cnn_v56 import FrugalEMGDataset, eval_raw, metrics, per_movement

DATA="/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
R="/mnt/data/medical_project/ninapro_db2_project/results"
OUT=f"{R}/v61"; os.makedirs(OUT, exist_ok=True)
CKPT=f"{OUT}/best_model_v61.pth"
CHDROP_Q=0.10   # 전극별 독립 드롭 확률(전극=raw i+env i+12 쌍). 평균 ~1.2개 드롭.


def aug_channel_drop(x, q=CHDROP_Q):
    B,C,T=x.shape; ne=C//2
    keep=(torch.rand(B,ne,device=x.device)>q).float()
    keep=torch.cat([keep,keep],dim=1).unsqueeze(-1)   # 전극 i와 i+12 동시
    return x*keep

def apply_aug_v61(x):
    if torch.rand(1).item()<0.5: x=aug_magnitude_warp(x,MAGWARP_SIGMA,WARP_KNOTS)
    if torch.rand(1).item()<0.5: x=aug_time_warp(x,TIMEWARP_SIGMA,WARP_KNOTS)
    if torch.rand(1).item()<0.5: x=aug_jitter(x,JITTER_SIGMA)
    x=aug_channel_drop(x)          # 매 배치 (q가 비율 조절)
    return x

def train_epoch_v61(model, loader, opt):
    model.train(); preds,labels=[],[]
    for xb,yb,sb in loader:
        xb,yb,sb=xb.to(DEVICE),yb.to(DEVICE),sb.to(DEVICE)
        xb=apply_aug_v61(xb)
        mix=torch.rand(1).item()<0.5
        if mix: xb,idx,lam=mixup_batch(xb,MIXUP_ALPHA); yb2=yb[idx]
        logits=model(xb,sb)
        loss=(lam*focal_loss(logits,yb,FOCAL_GAMMA)+(1-lam)*focal_loss(logits,yb2,FOCAL_GAMMA)) if mix else focal_loss(logits,yb,FOCAL_GAMMA)
        opt.zero_grad(); loss.backward(); opt.step()
        preds.extend(logits.detach().argmax(1).cpu().numpy()); labels.extend(yb.cpu().numpy())
    return accuracy_score(labels,preds)

@torch.no_grad()
def eval_drop(model, loader, elecs):
    """elecs: 0으로 만들 전극 리스트. 프레임 acc/hard 반환."""
    ch=np.concatenate([np.array(elecs),np.array(elecs)+12]) if len(elecs) else np.array([],dtype=int)
    model.eval(); preds,labels,subs=[],[],[]
    for xb,yb,sb in loader:
        xb=xb.to(DEVICE)
        if len(ch): xb[:,ch,:]=0.0
        preds.extend(model(xb,sb.to(DEVICE)).argmax(1).cpu().numpy()); labels.extend(yb.numpy()); subs.extend(sb.numpy())
    p,y,s=np.array(preds),np.array(labels),np.array(subs); h=np.isin(s,list(HARD_SUBJECTS))
    return float(accuracy_score(y,p)), float(accuracy_score(y[h],p[h]))

def drop_curve(model, loader, ncl):
    clean,_=eval_drop(model,loader,[])
    out={"clean":clean}
    for k in [1,2,3]:
        accs=[]
        for t in range(3):
            elec=np.random.RandomState(100+t).choice(12,size=k,replace=False)
            accs.append(eval_drop(model,loader,list(elec))[0])
        out[f"drop{k}"]=float(np.mean(accs))
    return out

def main():
    t0=time.time()
    print(f"Device {DEVICE} | v61 = v40 레시피 + 채널드롭(q={CHDROP_Q})")
    Xtr=np.load(f"{DATA}/X_train.npy"); ytr=np.load(f"{DATA}/y_train.npy"); str_=np.load(f"{DATA}/s_train.npy")
    Xv=np.load(f"{DATA}/X_val.npy"); yv=np.load(f"{DATA}/y_val.npy"); sv=np.load(f"{DATA}/s_val.npy")
    Xte=np.load(f"{DATA}/X_test.npy"); yte=np.load(f"{DATA}/y_test.npy"); ste=np.load(f"{DATA}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); ytr,yv,yte=le.transform(ytr),le.transform(yv),le.transform(yte)
    ncl=len(le.classes_); nc=Xtr.shape[2]; ns=int(str_.max())
    print(f"Train {len(ytr):,}  nc={nc} ncl={ncl} ns={ns}")
    tl=DataLoader(FrugalEMGDataset(Xtr,ytr,str_),batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(FrugalEMGDataset(Xv,yv,sv),batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)
    tel=DataLoader(FrugalEMGDataset(Xte,yte,ste),batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)

    model=SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=T_MAX,eta_min=1e-6)

    from train_cnn_v40 import evaluate
    best,bep,pc=-1,0,0
    print(f"{'Ep':>3} {'tr':>7} {'vacc':>7} {'vf1':>7} {'hard':>7}")
    for ep in range(1,N_EPOCHS+1):
        tr=train_epoch_v61(model,tl,opt); va,vf,vh,ve=evaluate(model,vl); sch.step()
        fl=""
        if vf>best: best,bep,pc=vf,ep,0; torch.save(model.state_dict(),CKPT); fl=" ↑"
        else:
            pc+=1
            if pc>=PATIENCE: print(f"Early stop @ {ep}"); break
        print(f"{ep:>3} {tr:>7.4f} {va:>7.4f} {vf:>7.4f} {vh:>7.4f}{fl}")

    print(f"\n[테스트] best ep={bep}")
    model.load_state_dict(torch.load(CKPT,map_location=DEVICE,weights_only=True))
    p,y,s=eval_raw(model,tel); fm=metrics(p,y,s,ncl); ip,iy,isb=per_movement(y,p,s,ncl); pm=metrics(ip,iy,isb,ncl)
    print(f"  clean 프레임 {fm[0]*100:.2f}%  동작단위 {pm[0]*100:.2f}%  hard {fm[2]*100:.2f}%")

    # 전극고장 곡선: v61 vs baseline v40
    print("\n[전극 고장 강건성] v61(채널드롭) vs v40(baseline)")
    v61c=drop_curve(model,tel,ncl)
    base=SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns).to(DEVICE)
    base.load_state_dict(torch.load(f"{R}/v40/best_model_v40.pth",map_location=DEVICE,weights_only=True))
    v40c=drop_curve(base,tel,ncl)
    print(f"  {'조건':<8}{'v40 base':>10}{'v61 chdrop':>12}{'개선':>8}")
    for key in ["clean","drop1","drop2","drop3"]:
        d=(v61c[key]-v40c[key])*100
        print(f"  {key:<8}{v40c[key]*100:>9.2f}%{v61c[key]*100:>11.2f}%{d:>+7.2f}p")

    json.dump({"model":"v61_channel_dropout_poc","chdrop_q":CHDROP_Q,"best_epoch":bep,
        "clean":{"frame":fm[0],"per_movement":pm[0],"hard":fm[2]},
        "drop_curve_v61":v61c,"drop_curve_v40_base":v40c,"elapsed_sec":time.time()-t0},
        open(f"{OUT}/result_v61.json","w"),indent=2)
    print(f"\n총 {(time.time()-t0)/60:.1f}분  저장: {OUT}/result_v61.json")

if __name__=="__main__": main()
