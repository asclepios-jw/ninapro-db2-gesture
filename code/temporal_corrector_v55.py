"""
v55: 시간 보정 모델(temporal corrector) — 손튜닝 평활/분할을 '학습'으로 대체.

동기: v51~53에서 프레임(73%)↔oracle(88%) 15p가 시간 구조에 잠김을 확인했고,
      손튜닝으로 80%까지 회수. 그러나 모델이 시간 구조를 '학습'한 적은 없음
      (모든 윈도우 독립 분류). 이 버전은 강한 앙상블의 윈도우 확률 스트림을
      2차 시퀀스 모델이 보고 일시적 rest-플립·경계를 학습으로 교정한다.

방법:
  1. 앙상블 8모델의 윈도우 확률을 val(rep5)·test(rep6)에서 계산.
     (멤버는 rep1-4로 gradient 학습, rep5는 선택에만 → rep5는 준-out-of-fold.)
  2. 피험자 시간순 시퀀스로 묶어, val 시퀀스로 코렉터(LSTM)를 학습:
     입력 = log(앙상블 확률) 시퀀스, 타깃 = 참 라벨 시퀀스, per-timestep CE.
  3. test에서 교정 예측 → 프레임 acc/macroF1/hard + per-movement 다수결.
  두 모드: bidir(오프라인, 미래 문맥) / causal(단방향, 실시간).

비교: 프레임 73.0 / 인과평활 74.7 / 중심평활 78.6 / 실제분할 80.4 / oracle 88.0.
caveat: rep5가 체크포인트 선택에 쓰여 약한 겹침. 참고 수치로 해석.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8  import SubjectAwareCNNLSTMAttention
from models_v41 import SubjectAwareResCNNLSTMAttention
from models_v44 import SubjectAwareCNNLSTMAttentionFeat
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

DATA_DIR="/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR ="/mnt/data/medical_project/ninapro_db2_project/results/v55"
os.makedirs(OUT_DIR,exist_ok=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS={28,31,4,26,7}; R="/mnt/data/medical_project/ninapro_db2_project/results"
ENSEMBLE=[
    (f"{R}/v44/best_model_v44.pth","v44"),(f"{R}/v44_rep1/best_model_v44.pth","v44"),
    (f"{R}/v44_rep2/best_model_v44.pth","v44"),(f"{R}/v45/best_model_v45.pth","v45"),
    (f"{R}/v45_rep2/best_model_v45.pth","v45"),(f"{R}/v40/best_model_v40.pth","v8"),
    (f"{R}/v41/best_model_v41.pth","v41"),(f"{R}/v38/best_model_v38.pth","v8"),
]
CHUNK=256; STRIDE=128; EPOCHS=40; LR=2e-3; NCL=50


class DS(Dataset):
    def __init__(s,X,y,sub):
        s.X=torch.from_numpy(X.transpose(0,2,1).astype(np.float32)); s.y=torch.from_numpy(y.astype(np.int64)); s.s=torch.from_numpy(sub.astype(np.int64))
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.X[i],s.y[i],s.s[i]

def build(k,nc,ncl,ns):
    if k=="v8":  return SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v41": return SubjectAwareResCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v44": return SubjectAwareCNNLSTMAttentionFeat(n_channels=nc,n_classes=ncl,n_subjects=ns)
    return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc,n_classes=ncl,n_subjects=ns)

@torch.no_grad()
def ens_probs(loader,nc,ncl,ns):
    acc=None
    for pth,k in ENSEMBLE:
        m=build(k,nc,ncl,ns).to(DEVICE); m.load_state_dict(torch.load(pth,map_location=DEVICE,weights_only=True)); m.eval()
        o=[]
        for xb,yb,sb in loader: o.append(F.softmax(m(xb.to(DEVICE),sb.to(DEVICE)),1).cpu().numpy())
        p=np.concatenate(o,0); acc=p if acc is None else acc+p
        del m; torch.cuda.empty_cache()
    return acc/len(ENSEMBLE)


class Corrector(nn.Module):
    def __init__(s,ncl=NCL,hid=64,layers=1,bidir=True):
        super().__init__()
        s.lstm=nn.LSTM(ncl,hid,num_layers=layers,batch_first=True,bidirectional=bidir,dropout=0.0)
        s.fc=nn.Linear(hid*(2 if bidir else 1),ncl)
    def forward(s,x):  # x:(B,L,ncl) log-prob
        h,_=s.lstm(x); return s.fc(h)


def seqs_by_subject(P, y, subs):
    """피험자별 (logprob_seq, label_seq) 리스트. 저장순서=시간순."""
    logP=np.log(P+1e-8).astype(np.float32)
    out=[]
    for s in np.unique(subs):
        idx=np.where(subs==s)[0]
        out.append((s, logP[idx], y[idx]))
    return out


def make_chunks(seqs):
    xs,ys=[],[]
    for _,lp,lb in seqs:
        L=len(lb)
        for a in range(0,max(1,L-CHUNK+1),STRIDE):
            b=min(a+CHUNK,L)
            xs.append(lp[a:b]); ys.append(lb[a:b])
        if L>CHUNK and (L-CHUNK)%STRIDE!=0:  # 마지막 꼬리
            xs.append(lp[L-CHUNK:L]); ys.append(lb[L-CHUNK:L])
    return xs,ys


def train_corrector(train_seqs, bidir):
    net=Corrector(bidir=bidir).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    xs,ys=make_chunks(train_seqs)
    idx=np.arange(len(xs))
    for ep in range(EPOCHS):
        np.random.shuffle(idx); net.train()
        for a in range(0,len(idx),32):
            bi=idx[a:a+32]
            xb=torch.stack([torch.from_numpy(xs[i]) for i in bi]).to(DEVICE)  # (B,CHUNK,ncl)
            yb=torch.stack([torch.from_numpy(ys[i]) for i in bi]).to(DEVICE)
            logits=net(xb)
            loss=F.cross_entropy(logits.reshape(-1,NCL),yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def eval_corrector(net, test_seqs):
    net.eval(); preds=[]; labels=[]; subs=[]
    for s,lp,lb in test_seqs:
        x=torch.from_numpy(lp).unsqueeze(0).to(DEVICE)   # (1,L,ncl)
        p=net(x).squeeze(0).argmax(1).cpu().numpy()
        preds.append(p); labels.append(lb); subs.append(np.full(len(lb),s))
    return np.concatenate(preds),np.concatenate(labels),np.concatenate(subs)


def met(y,p,subs):
    h=np.isin(subs,list(HARD_SUBJECTS))
    return (accuracy_score(y,p),f1_score(y,p,average="macro",zero_division=0,labels=range(NCL)),
            accuracy_score(y[h],p[h]))

def per_movement(y,p,subs):
    key=subs.astype(np.int64)*1000+y.astype(np.int64)
    ch=np.concatenate([[True],key[1:]!=key[:-1]]); ids=np.cumsum(ch)-1
    iy=[];ip=[];isb=[]
    for i in range(ids.max()+1):
        m=ids==i; iy.append(y[m][0]); isb.append(subs[m][0]); ip.append(np.bincount(p[m],minlength=NCL).argmax())
    iy,ip,isb=np.array(iy),np.array(ip),np.array(isb)
    return met(iy,ip,isb)


def main():
    t0=time.time()
    ytr=np.load(f"{DATA_DIR}/y_train.npy")
    Xv=np.load(f"{DATA_DIR}/X_val.npy"); yv=np.load(f"{DATA_DIR}/y_val.npy"); sv=np.load(f"{DATA_DIR}/s_val.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); yv=le.transform(yv); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA_DIR}/s_train.npy").max())

    print("앙상블 확률 계산 (val+test)...")
    vl=DataLoader(DS(Xv,yv,sv),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
    tl=DataLoader(DS(Xte,yte,ste),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
    Pv=ens_probs(vl,nc,ncl,ns); Pt=ens_probs(tl,nc,ncl,ns)

    # 기준: 교정 없는 앙상블 프레임(test)
    ba,bf,bh=met(yte,Pt.argmax(1),ste)
    print(f"\n[기준] 앙상블 프레임(test): acc {ba*100:.2f}% macroF1 {bf:.4f} hard {bh*100:.1f}%")
    print(f"[참조] 중심평활 78.6 / 실제분할 80.4 / oracle 88.0\n")

    train_seqs=seqs_by_subject(Pv,yv,sv)
    test_seqs =seqs_by_subject(Pt,yte,ste)

    res={}
    for mode,bidir in [("bidir(오프라인)",True),("causal(실시간)",False)]:
        net=train_corrector(train_seqs,bidir)
        p,y,s=eval_corrector(net,test_seqs)
        fa,ff,fh=met(y,p,s); ma,mf,mh=per_movement(y,p,s)
        print(f"[{mode}] 프레임: acc {fa*100:.2f}% ({(fa-ba)*100:+.2f}p) macroF1 {ff:.4f} hard {fh*100:.1f}%")
        print(f"           per-movement: acc {ma*100:.2f}% macroF1 {mf:.4f} hard {mh*100:.1f}%")
        res[mode]={"frame":{"acc":fa,"macro_f1":ff,"hard":fh},"per_movement":{"acc":ma,"macro_f1":mf,"hard":mh}}

    print("\n"+"═"*66)
    print("[v55 판정 — 학습된 시간 보정]")
    print(f"  기준 프레임 73.00% → bidir 프레임 {res['bidir(오프라인)']['frame']['acc']*100:.2f}% / "
          f"causal 프레임 {res['causal(실시간)']['frame']['acc']*100:.2f}%")
    print(f"  손튜닝 대비: 실제분할 80.44% vs 학습 per-movement "
          f"{res['bidir(오프라인)']['per_movement']['acc']*100:.2f}%(bidir)")
    print(f"  총 {time.time()-t0:.0f}s")
    json.dump({"model":"v55_temporal_corrector","baseline_frame":{"acc":ba,"macro_f1":bf,"hard":bh},
               "refs":{"centered":0.7864,"realseg":0.8044,"oracle":0.8804},"results":res},
              open(f"{OUT_DIR}/result_v55.json","w"),indent=2)
    print(f"저장: {OUT_DIR}/result_v55.json")

if __name__=="__main__": main()
