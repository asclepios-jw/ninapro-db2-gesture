"""
v34: v28 전처리 + v8 arch + hard subject oversampling (2×).

목적: s04, s07처럼 fine-tuning에서도 일관되게 악화되는 hard subjects를
      pooled 학습 단계에서 해결. train set에서 hard subjects 2× 오버샘플링.

v32 대비 변경:
  - train set: hard subjects {4,7,26,28,31} 윈도우를 2배로 복제 후 학습
  - 나머지 모두 동일: v28 전처리, v8 arch, AdamW 1e-3, CosineAnnealingLR, 50ep

비교 기준:
  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)
  v33  68.89%  v28 전처리 + v8 arch (fine-tuned)
  v32  68.70%  v28 전처리 + v8 arch (pooled, oversampling 없음)
  v34     ?    v28 전처리 + v8 arch (hard ×2 oversampling)

기대: hard acc 57.84%(v32) → 60%+ / overall 68.70% + 0.3~0.5p
데이터: 전처리전달_v28/
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from models_v8 import SubjectAwareCNNLSTMAttention

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v28"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v34"
CKPT     = os.path.join(OUT_DIR, "best_model_v34.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS      = 50
PATIENCE      = 8
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
OVERSAMPLE    = 2        # hard subjects 복제 배수
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}   # 1-indexed


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.s[i]


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    preds, labels = [], []
    for xb, yb, sb in loader:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        logits = model(xb, sb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        preds.extend(logits.detach().argmax(1).cpu().numpy())
        labels.extend(yb.cpu().numpy())
    return accuracy_score(labels, preds)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels, subs = [], [], []
    for xb, yb, sb in loader:
        xb, sb = xb.to(DEVICE), sb.to(DEVICE)
        preds.extend(model(xb, sb).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
        subs.extend(sb.cpu().numpy())
    preds  = np.array(preds)
    labels = np.array(labels)
    subs   = np.array(subs)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    hard_idx = np.isin(subs, list(HARD_SUBJECTS))
    hard_acc = accuracy_score(labels[hard_idx],  preds[hard_idx])
    easy_acc = accuracy_score(labels[~hard_idx], preds[~hard_idx])
    return acc, macro_f1, hard_acc, easy_acc


def oversample_hard(X, y, s, hard_subjects, factor):
    """hard subjects 윈도우를 factor배로 복제하여 train set에 추가."""
    hard_mask = np.isin(s, list(hard_subjects))
    n_hard = hard_mask.sum()
    X_aug = np.concatenate([X] + [X[hard_mask]] * (factor - 1), axis=0)
    y_aug = np.concatenate([y] + [y[hard_mask]] * (factor - 1), axis=0)
    s_aug = np.concatenate([s] + [s[hard_mask]] * (factor - 1), axis=0)
    print(f"  오버샘플링: hard {n_hard:,}개 × {factor} → "
          f"총 {len(X_aug):,}개 (원본 {len(X):,}개 + 추가 {len(X_aug)-len(X):,}개)")
    return X_aug, y_aug, s_aug


def main():
    print(f"Device: {DEVICE}")
    print(f"Hard subjects: {sorted(HARD_SUBJECTS)}  ×{OVERSAMPLE}")
    t0 = time.time()

    print("\n데이터 로드 중...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val   = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test  = np.load(os.path.join(DATA_DIR, "s_test.npy"))

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)

    # Hard subject 오버샘플링 (train only)
    X_train, y_train, s_train = oversample_hard(
        X_train, y_train, s_train, HARD_SUBJECTS, OVERSAMPLE
    )

    n_classes  = len(le.classes_)
    n_channels = X_train.shape[2]
    n_subjects = int(s_train.max())

    print(f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")
    print(f"n_channels={n_channels}  n_classes={n_classes}  n_subjects={n_subjects}")
    print(f"X_train mean={X_train.mean():.4f}  std={X_train.std():.4f}\n")

    train_loader = DataLoader(EMGDataset(X_train, y_train, s_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EMGDataset(X_val,   y_val,   s_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(EMGDataset(X_test,  y_test,  s_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SubjectAwareCNNLSTMAttention(
        n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"아키텍처: SubjectAwareCNNLSTMAttention (v8)")
    print(f"파라미터 수: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6
    )

    best_val_f1  = -1.0
    patience_ctr = 0
    history      = []

    print(f"{'Ep':>3}  {'tr_acc':>7}  {'val_acc':>7}  {'val_f1':>7}  "
          f"{'hard':>7}  {'easy':>7}")

    for epoch in range(1, N_EPOCHS + 1):
        tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_f1, val_hard, val_easy = evaluate(model, val_loader)
        scheduler.step()

        flag = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CKPT)
            patience_ctr = 0
            flag = " ↑"
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

        print(f"{epoch:>3}  {tr_acc:>7.4f}  {val_acc:>7.4f}  {val_f1:>7.4f}  "
              f"{val_hard:>7.4f}  {val_easy:>7.4f}{flag}")

        history.append({"epoch": epoch, "train_acc": tr_acc,
                        "val_acc": val_acc, "val_macro_f1": val_f1,
                        "val_hard_acc": val_hard, "val_easy_acc": val_easy})

    print("\n[테스트 평가]")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    test_acc, test_f1, test_hard, test_easy = evaluate(model, test_loader)
    gap     = test_easy - test_hard
    elapsed = time.time() - t0

    print(f"Overall acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Macro-F1     : {test_f1:.4f}")
    print(f"Hard acc     : {test_hard:.4f}  ({test_hard*100:.2f}%)")
    print(f"Easy acc     : {test_easy:.4f}  ({test_easy*100:.2f}%)")
    print(f"Gap          : {gap:.4f}  ({gap*100:.2f}p)")
    print(f"\n비교:")
    print(f"  v10  69.10%  v8 데이터 + v8 arch (fine-tuned)")
    print(f"  v33  68.89%  v28 전처리 + v8 arch (fine-tuned)")
    print(f"  v32  68.70%  v28 전처리 + v8 arch (oversampling 없음)")
    print(f"  v34  {test_acc*100:.2f}%  v28 전처리 + v8 arch (hard ×{OVERSAMPLE})")
    print(f"  hard: v32={57.84:.2f}% → v34={test_hard*100:.2f}%  "
          f"({(test_hard-0.5784)*100:+.2f}p)")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v34_v28preproc_v8arch_hardOS",
        "architecture": "SubjectAwareCNNLSTMAttention (models_v8.py)",
        "data": "전처리전달_v28",
        "oversample_factor": OVERSAMPLE,
        "hard_subjects": sorted(HARD_SUBJECTS),
        "overall_accuracy": test_acc,
        "overall_macro_f1": test_f1,
        "hard_acc": test_hard,
        "easy_acc": test_easy,
        "gap": gap,
        "best_val_macro_f1": best_val_f1,
        "elapsed_sec": elapsed,
        "baselines": {
            "v10_finetuned": {"overall_accuracy": 0.6910},
            "v33_finetuned": {"overall_accuracy": 0.6889},
            "v32_pooled":    {"overall_accuracy": 0.6870, "hard_acc": 0.5784},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v34.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v34.json")


if __name__ == "__main__":
    main()
