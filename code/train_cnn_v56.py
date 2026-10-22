"""
v56 (2/2): 자기지도 사전학습 CNN을 초기값으로 지도학습 finetune.

깔끔한 A/B 비교:
  v40 = 랜덤 초기화 CNN + (강한증강, focal, 50ep) 레시피 → pooled base ~70%
  v56 = SSL 대조학습 사전학습 CNN + 동일 레시피      → 이 스크립트
  차이는 오직 CNN 초기값(랜덤 vs 대조학습). SSL이 코어 프레임/hard를 올리는지 검증.

레시피(증강/focal/train/eval)는 train_cnn_v40에서 그대로 import → 완전 동일.
추가: 테스트에서 프레임 + 동작단위(per-movement 다수결) 동시 평가.

선행: pretrain_v56.py 실행 → results/v56/pretrained_cnn_v56.pth 생성돼 있어야 함.
출력: results/v56/best_model_v56.pth, result_v56.json
"""
import os, sys, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention
from train_cnn_v40 import (train_epoch, evaluate,
                           N_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY, T_MAX,
                           DEVICE, HARD_SUBJECTS)
from torch.utils.data import Dataset


class FrugalEMGDataset(Dataset):
    """메모리 절약형: X는 mmap으로 두고 샘플 하나씩만 (600,24)->(24,600) 변환.
    train_cnn_v40.EMGDataset는 전체를 RAM에 float32로 복제(피크 2×)해 OOM 위험 →
    이 버전은 RAM을 거의 쓰지 않는다(디스크에서 배치별로 읽음). 산출 텐서 형태 동일."""
    def __init__(self, X, y, s):
        self.X = X                                   # mmap ndarray (N,600,24)
        self.y = y.astype(np.int64)
        self.s = s.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = np.ascontiguousarray(np.asarray(self.X[i]).transpose(1, 0), dtype=np.float32)  # (24,600)
        return torch.from_numpy(x), int(self.y[i]), int(self.s[i])

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v56"
CKPT     = os.path.join(OUT_DIR, "best_model_v56.pth")
CNN_CKPT = os.path.join(OUT_DIR, "pretrained_cnn_v56.pth")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def eval_raw(model, loader):
    """프레임 예측 원본 반환(배열 순서 보존; loader shuffle=False)."""
    model.eval(); preds, labels, subs = [], [], []
    for xb, yb, sb in loader:
        preds.extend(model(xb.to(DEVICE), sb.to(DEVICE)).argmax(1).cpu().numpy())
        labels.extend(yb.numpy()); subs.extend(sb.numpy())
    return np.array(preds), np.array(labels), np.array(subs)


def metrics(y, p, subs, ncl):
    h = np.isin(subs, list(HARD_SUBJECTS))
    return (accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0, labels=range(ncl)),
            accuracy_score(y[h], p[h]), accuracy_score(y[~h], p[~h]))


def per_movement(y, p, subs, ncl):
    """연속된 같은 (subject,label) 구간 = 한 동작 실행. 다수결로 채점."""
    key = subs.astype(np.int64) * 1000 + y.astype(np.int64)
    ch = np.concatenate([[True], key[1:] != key[:-1]]); ids = np.cumsum(ch) - 1
    iy, ip, isb = [], [], []
    for i in range(ids.max() + 1):
        m = ids == i
        iy.append(y[m][0]); isb.append(subs[m][0]); ip.append(np.bincount(p[m], minlength=ncl).argmax())
    iy, ip, isb = np.array(iy), np.array(ip), np.array(isb)
    return metrics(iy, ip, isb, ncl)


def main():
    if not os.path.exists(CNN_CKPT):
        print(f"[에러] 사전학습 CNN 없음: {CNN_CKPT}\n먼저 pretrain_v56.py 를 실행하세요.")
        sys.exit(1)
    t0 = time.time()
    print(f"Device: {DEVICE} | v56 = SSL 사전학습 CNN + v40 레시피(finetune)")

    X_train = np.load(f"{DATA_DIR}/X_train.npy", mmap_mode="r"); y_train = np.load(f"{DATA_DIR}/y_train.npy"); s_train = np.load(f"{DATA_DIR}/s_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy", mmap_mode="r");   y_val   = np.load(f"{DATA_DIR}/y_val.npy");   s_val   = np.load(f"{DATA_DIR}/s_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy", mmap_mode="r");  y_test  = np.load(f"{DATA_DIR}/y_test.npy");  s_test  = np.load(f"{DATA_DIR}/s_test.npy")
    le = LabelEncoder(); le.fit(y_train)
    y_train, y_val, y_test = le.transform(y_train), le.transform(y_val), le.transform(y_test)
    ncl = len(le.classes_); nch = X_train.shape[2]; ns = int(s_train.max())
    print(f"Train {len(X_train):,}  n_channels={nch} n_classes={ncl} n_subjects={ns}")

    train_loader = DataLoader(FrugalEMGDataset(X_train, y_train, s_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(FrugalEMGDataset(X_val,   y_val,   s_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(FrugalEMGDataset(X_test,  y_test,  s_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SubjectAwareCNNLSTMAttention(n_channels=nch, n_classes=ncl, n_subjects=ns).to(DEVICE)
    # ★ SSL 사전학습된 CNN 가중치 로드 (나머지는 랜덤 초기화)
    cnn_state = torch.load(CNN_CKPT, map_location=DEVICE, weights_only=True)
    model.cnn.load_state_dict(cnn_state)
    print(f"사전학습 CNN 로드 완료: {CNN_CKPT}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=1e-6)

    best_f1, best_ep, pc, hist = -1.0, 0, 0, []
    print(f"{'Ep':>3} {'tr_acc':>7} {'val_acc':>7} {'val_f1':>7} {'hard':>7} {'easy':>7}")
    for ep in range(1, N_EPOCHS + 1):
        tr = train_epoch(model, train_loader, opt)
        va, vf, vh, ve = evaluate(model, val_loader)
        sched.step()
        flag = ""
        if vf > best_f1:
            best_f1, best_ep, pc = vf, ep, 0
            torch.save(model.state_dict(), CKPT); flag = " ↑"
        else:
            pc += 1
            if pc >= PATIENCE: print(f"Early stop @ {ep}"); break
        print(f"{ep:>3} {tr:>7.4f} {va:>7.4f} {vf:>7.4f} {vh:>7.4f} {ve:>7.4f}{flag}")
        hist.append({"epoch": ep, "train_acc": tr, "val_acc": va, "val_macro_f1": vf, "val_hard": vh, "val_easy": ve})

    print(f"\n[테스트]  best epoch={best_ep}")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    p, y, s = eval_raw(model, test_loader)
    fa, ff, fh, fe = metrics(y, p, s, ncl)
    ma, mf, mh, me = per_movement(y, p, s, ncl)
    elapsed = time.time() - t0

    print(f"\n── 프레임(윈도우) ──")
    print(f"  acc {fa*100:.2f}%  macroF1 {ff:.4f}  hard {fh*100:.2f}%  easy {fe*100:.2f}%  gap {(fe-fh)*100:.2f}p")
    print(f"── 동작 단위(per-movement 다수결) ──")
    print(f"  acc {ma*100:.2f}%  macroF1 {mf:.4f}  hard {mh*100:.2f}%  easy {me*100:.2f}%")
    print(f"\n[비교] v40 pooled base ~70.3% (랜덤 CNN) 대비 v56(SSL CNN) 프레임 {fa*100:.2f}% ({(fa-0.703)*100:+.2f}p)")
    print(f"총 {elapsed/60:.1f}분")

    json.dump({"model": "v56_ssl_pretrain_finetune",
               "note": "SSL contrastive-pretrained CNN + v40 recipe",
               "frame": {"acc": fa, "macro_f1": ff, "hard": fh, "easy": fe},
               "per_movement": {"acc": ma, "macro_f1": mf, "hard": mh, "easy": me},
               "best_epoch": best_ep, "best_val_macro_f1": best_f1,
               "baseline_v40_pooled_frame": 0.703, "elapsed_sec": elapsed, "history": hist},
              open(f"{OUT_DIR}/result_v56.json", "w"), indent=2)
    print(f"저장: {OUT_DIR}/result_v56.json")


if __name__ == "__main__":
    main()
