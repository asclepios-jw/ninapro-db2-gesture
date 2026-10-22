"""v61 평가(버그 수정본): 저장된 best_model_v61 로드 → clean + 전극고장 곡선, v40 비교."""
import os, sys, json, time
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention
from train_cnn_v56 import FrugalEMGDataset, eval_raw, metrics, per_movement
from train_cnn_v61 import drop_curve

DATA="/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
R="/mnt/data/medical_project/ninapro_db2_project/results"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    t0=time.time()
    Xte=np.load(f"{DATA}/X_test.npy",mmap_mode="r"); yte=np.load(f"{DATA}/y_test.npy"); ste=np.load(f"{DATA}/s_test.npy")
    ytr=np.load(f"{DATA}/y_train.npy"); le=LabelEncoder(); le.fit(ytr); yte=le.transform(yte)
    ncl=len(le.classes_); nc=Xte.shape[2]; ns=int(np.load(f"{DATA}/s_train.npy").max())
    tel=DataLoader(FrugalEMGDataset(Xte,yte,ste),batch_size=256,shuffle=False,num_workers=3,pin_memory=True)

    v61=SubjectAwareCNNLSTMAttention(nc,ncl,ns).to(DEV)
    v61.load_state_dict(torch.load(f"{R}/v61/best_model_v61.pth",map_location=DEV,weights_only=True))
    p,y,s=eval_raw(v61,tel)                      # (preds, labels, subs)
    fm=metrics(y,p,s,ncl); pm=per_movement(y,p,s,ncl)   # (acc,f1,hard,easy)
    print(f"[v61 clean] 프레임 {fm[0]*100:.2f}%  동작단위 {pm[0]*100:.2f}%  hard(프레임) {fm[2]*100:.2f}%")

    base=SubjectAwareCNNLSTMAttention(nc,ncl,ns).to(DEV)
    base.load_state_dict(torch.load(f"{R}/v40/best_model_v40.pth",map_location=DEV,weights_only=True))
    pb,yb,sb=eval_raw(base,tel); fmb=metrics(yb,pb,sb,ncl); pmb=per_movement(yb,pb,sb,ncl)
    print(f"[v40 clean] 프레임 {fmb[0]*100:.2f}%  동작단위 {pmb[0]*100:.2f}%  hard(프레임) {fmb[2]*100:.2f}%")

    print("\n[전극 고장 강건성]  (프레임 acc)")
    c61=drop_curve(v61,tel,ncl); c40=drop_curve(base,tel,ncl)
    print(f"  {'조건':<7}{'v40 base':>10}{'v61 chdrop':>12}{'개선':>9}")
    for k in ["clean","drop1","drop2","drop3"]:
        print(f"  {k:<7}{c40[k]*100:>9.2f}%{c61[k]*100:>11.2f}%{(c61[k]-c40[k])*100:>+8.2f}p")

    json.dump({"model":"v61_channel_dropout_poc","chdrop_q":0.10,
        "v61_clean":{"frame":fm[0],"per_movement":pm[0],"hard":fm[2]},
        "v40_clean":{"frame":fmb[0],"per_movement":pmb[0],"hard":fmb[2]},
        "drop_curve_v61":c61,"drop_curve_v40_base":c40,"elapsed_sec":time.time()-t0},
        open(f"{R}/v61/result_v61.json","w"),indent=2)
    print(f"\n총 {time.time()-t0:.0f}s  저장: {R}/v61/result_v61.json")

if __name__=="__main__": main()
