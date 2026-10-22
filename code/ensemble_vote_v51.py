"""
v51: 반복(제스처 실행) 단위 다수결 평가 — 윈도우 레벨 macro-F1이 착시인지 검증.

MiniMax 리뷰 A3: 윈도우 레벨 macro-F1(0.644)은 경계 윈도우 노이즈로 과소평가
                일 수 있음. Ninapro 표준인 '반복 단위(제스처 실행) 다수결'로
                평가하면 gesture 레벨 성능이 나옴.

방법 (모델 불변, 추론 후처리만):
  1. v48 최적 8모델 앙상블의 윈도우 예측(확률 평균 argmax) 계산.
  2. test(rep6)를 저장 순서(피험자 내 시간순)대로 순회하며, 연속된 동일
     (subject, label) 런 = 하나의 제스처 실행 인스턴스로 묶음.
  3. 인스턴스별 윈도우 예측을 다수결 → 인스턴스 1개당 예측 1개.
  4. 인스턴스 레벨 acc / macro-F1 / hard·easy 계산, 윈도우 레벨과 비교.
     (rest 포함/제외 둘 다 보고.)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8  import SubjectAwareCNNLSTMAttention
from models_v41 import SubjectAwareResCNNLSTMAttention
from models_v44 import SubjectAwareCNNLSTMAttentionFeat
from models_v45 import SubjectAwareCNNLSTMAttentionFeatFD

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v51"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}
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


def build(k,nc,ncl,ns):
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


def metrics(y, p, subs, ncl, tag):
    acc=accuracy_score(y,p); mf1=f1_score(y,p,average="macro",zero_division=0,labels=range(ncl))
    hidx=np.isin(subs,list(HARD_SUBJECTS))
    hard=accuracy_score(y[hidx],p[hidx]); easy=accuracy_score(y[~hidx],p[~hidx])
    # 제스처(비-rest, label!=0)만의 macro-F1
    nz=y!=0
    mf1_g=f1_score(y[nz],p[nz],average="macro",zero_division=0)
    print(f"  [{tag}] acc {acc*100:.2f}%  macroF1 {mf1:.4f}  (제스처만 macroF1 {mf1_g:.4f})  hard {hard*100:.2f}%  easy {easy*100:.2f}%")
    return {"acc":acc,"macro_f1":mf1,"macro_f1_gesture":mf1_g,"hard":hard,"easy":easy}


def main():
    print("데이터 로드 (test rep6, 저장순서=피험자내 시간순 유지)...")
    ytr=np.load(f"{DATA_DIR}/y_train.npy")
    Xte=np.load(f"{DATA_DIR}/X_test.npy"); yte=np.load(f"{DATA_DIR}/y_test.npy"); ste=np.load(f"{DATA_DIR}/s_test.npy")
    le=LabelEncoder(); le.fit(ytr); y=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA_DIR}/s_train.npy").max())
    loader=DataLoader(DS(Xte,y,ste),batch_size=256,shuffle=False,num_workers=2,pin_memory=True)

    print(f"앙상블 8모델 윈도우 예측 계산...  (윈도우 {len(y):,}개)")
    P=np.mean([probs(p,k,loader,nc,ncl,ns) for p,k in ENSEMBLE],axis=0)
    win_pred=P.argmax(1)

    print("\n[윈도우 레벨] (기존 v48 방식)")
    win_m = metrics(y, win_pred, ste, ncl, "window")

    # 인스턴스 그룹핑: 연속된 동일 (subject, label) 런
    key = ste.astype(np.int64) * 1000 + y.astype(np.int64)   # (subject,label) 결합키
    change = np.concatenate([[True], key[1:] != key[:-1]])
    inst_id = np.cumsum(change) - 1
    n_inst = inst_id.max() + 1

    inst_y  = np.zeros(n_inst, dtype=np.int64)
    inst_s  = np.zeros(n_inst, dtype=np.int64)
    inst_p  = np.zeros(n_inst, dtype=np.int64)
    win_per_inst = np.zeros(n_inst, dtype=np.int64)
    for i in range(n_inst):
        m = inst_id == i
        inst_y[i] = y[m][0]; inst_s[i] = ste[m][0]
        inst_p[i] = np.bincount(win_pred[m], minlength=ncl).argmax()   # 다수결
        win_per_inst[i] = m.sum()

    print(f"\n[반복(제스처 실행) 단위 다수결]  인스턴스 {n_inst:,}개 "
          f"(평균 {win_per_inst.mean():.1f} 윈도우/인스턴스)")
    inst_m = metrics(inst_y, inst_p, inst_s, ncl, "instance")

    print("\n" + "═"*70)
    print("[v51 판정 — 윈도우 레벨 → 반복 단위]")
    print(f"  acc      : {win_m['acc']*100:.2f}% → {inst_m['acc']*100:.2f}%  ({(inst_m['acc']-win_m['acc'])*100:+.2f}p)")
    print(f"  macro-F1 : {win_m['macro_f1']:.4f} → {inst_m['macro_f1']:.4f}  ({inst_m['macro_f1']-win_m['macro_f1']:+.4f})")
    print(f"  제스처 macroF1: {win_m['macro_f1_gesture']:.4f} → {inst_m['macro_f1_gesture']:.4f}  ({inst_m['macro_f1_gesture']-win_m['macro_f1_gesture']:+.4f})")
    print(f"  hard acc : {win_m['hard']*100:.2f}% → {inst_m['hard']*100:.2f}%  ({(inst_m['hard']-win_m['hard'])*100:+.2f}p)")
    print(f"\n해석: macro-F1이 크게 오르면 '윈도우 레벨 착시'(경계 노이즈), "
          f"거의 그대로면 '실제 판별력 한계'(MiniMax A3 검증).")

    json.dump({"model":"v51_repetition_vote","n_instances":int(n_inst),
               "window_level":win_m,"instance_level":inst_m,
               "mean_windows_per_instance":float(win_per_inst.mean())},
              open(f"{OUT_DIR}/result_v51.json","w"),indent=2)
    print(f"\n저장: {OUT_DIR}/result_v51.json")


if __name__=="__main__":
    main()
