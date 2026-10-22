"""
채널드롭 재훈련 (범용) — v62(v44)·v63(v45)에 재사용.
환경변수: CHDROP_MODEL ∈ {v8,v44,v45,v41}, CHDROP_OUT (결과 디렉토리).
v40 레시피 + 채널드롭 증강(q=0.10). 모델 종류만 다르고 나머지 동일.
"""
import os, sys, time, json
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from stability_v57 import build
from train_cnn_v40 import (N_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY, T_MAX, DEVICE, evaluate)
from train_cnn_v56 import FrugalEMGDataset
from train_cnn_v61 import train_epoch_v61

DATA="/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
MODEL=os.environ.get("CHDROP_MODEL","v44")
OUT=os.environ["CHDROP_OUT"]; os.makedirs(OUT, exist_ok=True)
CKPT=f"{OUT}/best_model.pth"

def main():
    t0=time.time(); print(f"Device {DEVICE} | 채널드롭 재훈련 MODEL={MODEL} → {OUT}")
    Xtr=np.load(f"{DATA}/X_train.npy"); ytr=np.load(f"{DATA}/y_train.npy"); str_=np.load(f"{DATA}/s_train.npy")
    Xv=np.load(f"{DATA}/X_val.npy"); yv=np.load(f"{DATA}/y_val.npy"); sv=np.load(f"{DATA}/s_val.npy")
    le=LabelEncoder(); le.fit(ytr); ytr,yv=le.transform(ytr),le.transform(yv)
    ncl=len(le.classes_); nc=Xtr.shape[2]; ns=int(str_.max())
    print(f"Train {len(ytr):,} nc={nc} ncl={ncl} ns={ns}")
    tl=DataLoader(FrugalEMGDataset(Xtr,ytr,str_),batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(FrugalEMGDataset(Xv,yv,sv),batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)

    model=build(MODEL,nc,ncl,ns).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=T_MAX,eta_min=1e-6)
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
    json.dump({"model":f"chdrop_{MODEL}","best_epoch":bep,"best_val_f1":best,"elapsed_sec":time.time()-t0},
              open(f"{OUT}/train_meta.json","w"),indent=2)
    print(f"\nbest ep={bep} val_f1={best:.4f}  {(time.time()-t0)/60:.1f}분  저장: {CKPT}")

if __name__=="__main__": main()
