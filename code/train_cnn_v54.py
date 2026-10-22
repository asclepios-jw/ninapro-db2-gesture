"""
v54: 스펙트로그램 2D-CNN pooled 학습 → 앙상블 새 멤버 (근본적 다른 아키텍처).

파이프라인은 v40/v44와 동일(강한 증강 + focal, 50ep). 모델만 SpectrogramCNN.
증강은 시간영역 x에 적용 후 모델 내부에서 STFT → 일관.
데이터: 전처리전달_v28/
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v54 import SpectrogramCNN

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v54"
CKPT     = os.path.join(OUT_DIR, "best_model_v54.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS=50; PATIENCE=8; BATCH_SIZE=256; LR=1e-3; WEIGHT_DECAY=1e-4; T_MAX=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS={28,31,4,26,7}
JITTER_P,JITTER_SIGMA=0.5,0.06
MAGWARP_P,MAGWARP_SIGMA=0.5,0.30
TIMEWARP_P,TIMEWARP_SIGMA=0.5,0.25
WARP_KNOTS=4
MIXUP_P,MIXUP_ALPHA=0.5,0.30
FOCAL_GAMMA=2.0


class EMGDataset(Dataset):
    def __init__(s,X,y,sub):
        s.X=torch.from_numpy(X.transpose(0,2,1).astype(np.float32)); s.y=torch.from_numpy(y.astype(np.int64)); s.s=torch.from_numpy(sub.astype(np.int64))
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.X[i],s.y[i],s.s[i]


def focal_loss(logits,target,g=2.0):
    lp=F.log_softmax(logits,1); lpt=lp.gather(1,target.unsqueeze(1)).squeeze(1)
    return (-((1-lpt.exp())**g)*lpt).mean()

def aug_jitter(x,s): return x+torch.randn_like(x)*s
def aug_magwarp(x,sigma,k):
    B,C,T=x.shape; kv=torch.randn(B,C//2,k,device=x.device)*sigma+1.0
    cu=F.interpolate(kv,size=T,mode="linear",align_corners=True); return x*torch.cat([cu,cu],1)
def aug_timewarp(x,sigma,k):
    B,C,T=x.shape; sp=torch.exp(torch.randn(B,1,k,device=x.device)*sigma)
    sp=F.interpolate(sp,size=T,mode="linear",align_corners=True)
    cum=torch.cumsum(sp,2); cum=cum-cum[:,:,:1]; cum=cum/cum[:,:,-1:]
    grid=torch.zeros(B,1,T,2,device=x.device); grid[...,0]=cum*2-1
    return F.grid_sample(x.unsqueeze(2),grid,mode="bilinear",align_corners=True,padding_mode="border").squeeze(2)
def apply_augment(x):
    if torch.rand(1).item()<MAGWARP_P:  x=aug_magwarp(x,MAGWARP_SIGMA,WARP_KNOTS)
    if torch.rand(1).item()<TIMEWARP_P: x=aug_timewarp(x,TIMEWARP_SIGMA,WARP_KNOTS)
    if torch.rand(1).item()<JITTER_P:   x=aug_jitter(x,JITTER_SIGMA)
    return x
def mixup(x,a):
    lam=float(np.random.beta(a,a)); idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx],idx,lam


def train_epoch(model,loader,opt):
    model.train(); pr,lb=[],[]
    for xb,yb,sb in loader:
        xb,yb,sb=xb.to(DEVICE),yb.to(DEVICE),sb.to(DEVICE)
        xb=apply_augment(xb)
        um=torch.rand(1).item()<MIXUP_P
        if um: xb,idx,lam=mixup(xb,MIXUP_ALPHA); yb2=yb[idx]
        logits=model(xb,sb)
        loss=(lam*focal_loss(logits,yb,FOCAL_GAMMA)+(1-lam)*focal_loss(logits,yb2,FOCAL_GAMMA)) if um else focal_loss(logits,yb,FOCAL_GAMMA)
        opt.zero_grad(); loss.backward(); opt.step()
        pr.extend(logits.detach().argmax(1).cpu().numpy()); lb.extend(yb.cpu().numpy())
    return accuracy_score(lb,pr)


@torch.no_grad()
def evaluate(model,loader):
    model.eval(); pr,lb,su=[],[],[]
    for xb,yb,sb in loader:
        pr.extend(model(xb.to(DEVICE),sb.to(DEVICE)).argmax(1).cpu().numpy()); lb.extend(yb.numpy()); su.extend(sb.numpy())
    pr,lb,su=np.array(pr),np.array(lb),np.array(su)
    hidx=np.isin(su,list(HARD_SUBJECTS))
    return accuracy_score(lb,pr),f1_score(lb,pr,average="macro",zero_division=0),accuracy_score(lb[hidx],pr[hidx]),accuracy_score(lb[~hidx],pr[~hidx])


def main():
    print(f"Device:{DEVICE}\n아키텍처: SpectrogramCNN (STFT n_fft=64,hop=16 → 2D-CNN)")
    t0=time.time(); print("데이터 로드...")
    Xtr=np.load(f"{DATA_DIR}/X_train.npy"); ytr=np.load(f"{DATA_DIR}/y_train.npy"); str_=np.load(f"{DATA_DIR}/s_train.npy")
    Xv=np.load(f"{DATA_DIR}/X_val.npy"); yv=np.load(f"{DATA_DIR}/y_val.npy"); sv=np.load(f"{DATA_DIR}/s_val.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); ytr=le.transform(ytr); yv=le.transform(yv); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xtr.shape[2]; ns=int(str_.max())
    print(f"Train {len(Xtr):,} Val {len(Xv):,} Test {len(Xte):,}  n_classes={ncl}\n")
    tl=DataLoader(EMGDataset(Xtr,ytr,str_),batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(EMGDataset(Xv,yv,sv),batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)
    tel=DataLoader(EMGDataset(Xte,yte,ste),batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)

    model=SpectrogramCNN(n_channels=nc,n_classes=ncl,n_subjects=ns).to(DEVICE)
    print(f"파라미터: {sum(p.numel() for p in model.parameters()):,}")
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=T_MAX,eta_min=1e-6)

    best=-1; best_ep=0; pc=0; hist=[]
    print(f"{'Ep':>3} {'tr':>7} {'vacc':>7} {'vf1':>7} {'hard':>6}")
    for ep in range(1,N_EPOCHS+1):
        tr=train_epoch(model,tl,opt); va,vf,vh,ve=evaluate(model,vl); sch.step()
        fl=""
        if vf>best: best=vf; best_ep=ep; torch.save(model.state_dict(),CKPT); pc=0; fl=" ↑"
        else:
            pc+=1
            if pc>=PATIENCE: print(f"Early stop ep{ep}"); break
        print(f"{ep:>3} {tr:>7.4f} {va:>7.4f} {vf:>7.4f} {vh:>6.3f}{fl}")
        hist.append({"epoch":ep,"val_acc":va,"val_macro_f1":vf})

    print(f"\n[테스트] best epoch={best_ep}")
    model.load_state_dict(torch.load(CKPT,map_location=DEVICE,weights_only=True))
    acc,f1,hard,easy=evaluate(model,tel); el=time.time()-t0
    print(f"Overall acc: {acc*100:.2f}%  Macro-F1: {f1:.4f}  Hard: {hard*100:.2f}%  Easy: {easy*100:.2f}%")
    print(f"\n단독 성능 참고: 기존 단일 특징모델 ~70.3%. (앙상블 멤버로는 약해도 비상관이면 기여 — v47 교훈)")
    print(f"총 시간: {el/60:.1f}분\n→ 다음: 앙상블에 추가해 재-sweep")
    json.dump({"model":"v54_spectrogram_cnn","architecture":"SpectrogramCNN",
               "n_params":int(sum(p.numel() for p in model.parameters())),
               "overall_accuracy":acc,"overall_macro_f1":f1,"hard_acc":hard,"easy_acc":easy,
               "best_epoch":best_ep,"elapsed_sec":el,"history":hist},
              open(f"{OUT_DIR}/result_v54_pooled.json","w"),indent=2)
    print(f"저장: {OUT_DIR}/result_v54_pooled.json")


if __name__=="__main__":
    main()
