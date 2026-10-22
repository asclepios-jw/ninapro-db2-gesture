"""
v30: 팀 데이터(10Hz/4차) + v8 실제 아키텍처 (models_v8.SubjectAwareCNNLSTMAttention).

목적: v25-v29에서 사용한 단순화 아키텍처(k=3, ReLU, MaxPool 없음)가 아닌
      실제 v8 아키텍처(k=5, ELU, MaxPool×2, Subject emb→LSTM 입력)로
      팀 데이터에서 성능을 측정 → 공정한 비교 기준선 확보.

v8 아키텍처 vs v25-v29 아키텍처 핵심 차이:
  - Conv k=5+ELU+MaxPool×2 (600→150 time steps) vs k=3+ReLU+NoMaxPool
  - Subject embedding을 LSTM 입력에 연결 vs LSTM 출력 후 연결
  - 비선형 Attention(tanh+linear) vs 선형 Attention(softmax)

비교 기준:
  v8   v8 데이터 + v8 아키텍처 (pooled)     : 68.68%
  v10  v8 데이터 + v8 아키텍처 (fine-tuned) : 69.10%
  v27  팀 10Hz 데이터 + 단순 아키텍처        : 63.56%
  v29  v8 전처리 데이터 + 단순 아키텍처      : 63.57%
  v30  팀 10Hz 데이터 + v8 아키텍처          : ?

데이터: 전처리전달_10hz/ (build_dataset_new.py, 이미 per-subject 정규화 완료)
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

DATA_DIR = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_10hz"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/results/v30"
CKPT     = os.path.join(OUT_DIR, "best_model_v30.pth")
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS     = 50
PATIENCE     = 8
BATCH_SIZE   = 256   # 512 OOM (RTX 2070 8GB)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_SUBJECTS = {28, 31, 4, 26, 7}


class EMGDataset(Dataset):
    def __init__(self, X, y, s):
        # 데이터 이미 정규화됨 → 추가 정규화 없음
        self.X = torch.from_numpy(X.transpose(0, 2, 1).astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.int64))   # 1-indexed (v8 방식)

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
    hard_idx = np.isin(subs, list(HARD_SUBJECTS))   # 1-indexed 그대로
    hard_acc = accuracy_score(labels[hard_idx],  preds[hard_idx])
    easy_acc = accuracy_score(labels[~hard_idx], preds[~hard_idx])
    return acc, macro_f1, hard_acc, easy_acc


def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()

    print("Loading data...")
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

    n_classes  = len(le.classes_)
    n_channels = X_train.shape[2]
    n_subjects = int(s_train.max())   # 40 (s는 1-indexed)

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

    # v8 실제 아키텍처: Embedding(n_subjects+1=41), subject emb → LSTM 입력
    model = SubjectAwareCNNLSTMAttention(
        n_channels=n_channels, n_classes=n_classes, n_subjects=n_subjects
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"아키텍처: SubjectAwareCNNLSTMAttention (v8 실제)")
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
    print(f"  v8   v8 데이터 + v8 아키텍처 (pooled)     : 68.68%")
    print(f"  v10  v8 데이터 + v8 아키텍처 (fine-tuned) : 69.10%")
    print(f"  v27  팀 10Hz 데이터 + 단순 아키텍처        : 63.56%")
    print(f"  v29  v8 전처리 데이터 + 단순 아키텍처      : 63.57%")
    print(f"  v30  팀 10Hz 데이터 + v8 아키텍처          : {test_acc*100:.2f}%")
    print(f"총 소요 시간: {elapsed/60:.1f}분")

    result = {
        "model": "v30_team10hz_v8arch",
        "architecture": "SubjectAwareCNNLSTMAttention (models_v8.py)",
        "data": "전처리전달_10hz (build_dataset_new.py, 10Hz/4차)",
        "overall_accuracy": test_acc,
        "overall_macro_f1": test_f1,
        "hard_acc": test_hard,
        "easy_acc": test_easy,
        "gap": gap,
        "best_val_macro_f1": best_val_f1,
        "elapsed_sec": elapsed,
        "baselines": {
            "v8_pooled":            {"overall_accuracy": 0.6868},
            "v10_finetuned":        {"overall_accuracy": 0.6910},
            "v27_team_simple_arch": {"overall_accuracy": 0.6356},
            "v29_v8data_simple_arch": {"overall_accuracy": 0.6357},
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result_v30.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"결과 저장: {OUT_DIR}/result_v30.json")


if __name__ == "__main__":
    main()
