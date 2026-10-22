"""
v49 진단: macro-F1이 낮은 이유 파악 — 클래스별 F1 + 혼동 (앙상블 예측 기반).

후처리 보정이 무효(v49)였으므로, 낮은 macro-F1은 판별력 문제. 어떤 클래스가
얼마나 낮고 무엇과 혼동되는지 보면 다음 처방(클래스균형 학습/특정쌍 등)이 잡힌다.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8  import SubjectAwareCNNLSTMAttention
from models_v41 import SubjectAwareResCNNLSTMAttention
from models_v44 import SubjectAwareCNNLSTMAttentionFeat
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = "/mnt/data/medical_project/ninapro_db2_project/results"
ENSEMBLE = [
    (f"{R}/v44/best_model_v44.pth","v44"),(f"{R}/v44_rep1/best_model_v44.pth","v44"),
    (f"{R}/v44_rep2/best_model_v44.pth","v44"),(f"{R}/v45/best_model_v45.pth","v45"),
    (f"{R}/v45_rep2/best_model_v45.pth","v45"),(f"{R}/v40/best_model_v40.pth","v8"),
    (f"{R}/v41/best_model_v41.pth","v41"),(f"{R}/v38/best_model_v38.pth","v8"),
]


class DS(Dataset):
    def __init__(s, X, y, sub):
        s.X=torch.from_numpy(X.transpose(0,2,1).astype(np.float32)); s.y=torch.from_numpy(y.astype(np.int64)); s.s=torch.from_numpy(sub.astype(np.int64))
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.X[i], s.y[i], s.s[i]


def build(k, nc, ncl, ns):
    if k=="v8":  return SubjectAwareCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v41": return SubjectAwareResCNNLSTMAttention(n_channels=nc,n_classes=ncl,n_subjects=ns)
    if k=="v44": return SubjectAwareCNNLSTMAttentionFeat(n_channels=nc,n_classes=ncl,n_subjects=ns)
    return SubjectAwareCNNLSTMAttentionFeatFD(n_channels=nc,n_classes=ncl,n_subjects=ns)


@torch.no_grad()
def probs(path,k,loader,nc,ncl,ns):
    m=build(k,nc,ncl,ns).to(DEVICE); m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True)); m.eval()
    o=[]
    for xb,yb,sb in loader: o.append(F.softmax(m(xb.to(DEVICE),sb.to(DEVICE)),1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(o,0)


def main():
    ytr=np.load(f"{DATA_DIR}/y_train.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); ty=le.transform(yte); ytr_e=le.transform(ytr)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader=DataLoader(DS(Xte,ty,ste),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    print("앙상블 예측 계산...")
    P=np.mean([probs(p,k,loader,nc,ncl,ns) for p,k in ENSEMBLE],axis=0)
    pred=P.argmax(1)
    print(f"전체 acc {accuracy_score(ty,pred)*100:.2f}%  macro-F1 {f1_score(ty,pred,average='macro',zero_division=0):.4f}\n")

    prec,rec,f1,sup = precision_recall_fscore_support(ty,pred,labels=range(ncl),zero_division=0)
    tr_cnt = np.bincount(ytr_e,minlength=ncl)  # train 지원(빈도)

    # F1 분포
    print(f"F1<0.5 클래스: {(f1<0.5).sum()}/{ncl}   F1<0.3: {(f1<0.3).sum()}   macro-F1={f1.mean():.4f}")
    print(f"가중 F1(빈도가중)={np.average(f1,weights=sup):.4f}  ← acc와 유사(다수클래스 잘함)\n")

    # 최악 12개 클래스: F1, test지원, train지원, 최다 혼동 클래스
    order=np.argsort(f1)[:12]
    cm_rows = {}
    for c in order:
        mask=(ty==c)
        others=pred[mask]; others=others[others!=c]
        conf = np.bincount(others,minlength=ncl).argmax() if len(others)>0 else -1
        cm_rows[c]=conf
    le_classes = le.classes_  # 원래 전역 라벨
    print(f"{'클래스(전역)':>10}  {'F1':>5}  {'prec':>5}  {'rec':>5}  {'test수':>6}  {'train수':>7}  {'최다혼동→(전역)':>14}")
    print("─"*72)
    for c in order:
        gl = le_classes[c]; conf_gl = le_classes[cm_rows[c]] if cm_rows[c]>=0 else -1
        print(f"{gl:>10}  {f1[c]:>5.2f}  {prec[c]:>5.2f}  {rec[c]:>5.2f}  {sup[c]:>6}  {tr_cnt[c]:>7}  {conf_gl:>14}")

    # 저F1과 지원(빈도) 상관: 낮은 F1이 희귀 클래스 때문인가?
    lowf1 = f1<0.5
    print(f"\nF1<0.5 클래스의 평균 train 지원: {tr_cnt[lowf1].mean():.0f}  vs  F1>=0.5: {tr_cnt[~lowf1].mean():.0f}")
    corr = np.corrcoef(np.log(tr_cnt+1), f1)[0,1]
    print(f"log(train빈도) vs F1 상관계수: {corr:+.3f}  (양수 클수록 '희귀=저F1')")

    os.makedirs(f"{R}/v49",exist_ok=True)
    json.dump({"per_class_f1":f1.tolist(),"support_test":sup.tolist(),"support_train":tr_cnt.tolist(),
               "n_f1_below_0.5":int((f1<0.5).sum()),"freq_f1_corr":float(corr)},
              open(f"{R}/v49/diagnose.json","w"),indent=2)
    print(f"\n저장: {R}/v49/diagnose.json")


if __name__=="__main__":
    main()
