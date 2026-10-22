"""
v10: v9에서 본 문제(전체 네트워크를 다 풀어서 fine-tuning하니 피험자별 분산이
평균 개선폭보다 훨씬 컸음, 일부는 오히려 악화) 해결 시도.
CNN 백본을 얼리고(freeze) classifier/attention 쪽만 학습하면 적은 데이터(피험자당
~11,000 윈도우)로도 과적합 없이 더 안정적으로 개인화될 것이라는 가설을 검증.

두 가지 freeze 모드를 v9(전체 fine-tune)와 비교:
  - freeze_cnn      : CNN 백본만 얼리고 subject_emb+LSTM+Attention+분류기는 학습
  - classifier_only : CNN+LSTM+Attention+subject_emb 전부 얼리고 분류기(fc)만 학습
"""
import os
import time
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from models_v8 import SubjectAwareCNNLSTMAttention
from finetune_v9 import EMGDataset, run_epoch, DATA_DIR, BASE_CKPT

OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v10"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FT_EPOCHS = 15
FT_PATIENCE = 5
FT_LR = 1e-4
FT_BATCH = 128

FREEZE_MODES = {
    "freeze_cnn": ["cnn"],
    "classifier_only": ["cnn", "lstm", "attention", "subject_emb"],
}


def apply_freeze(model, freeze_modules):
    for name, module in model.named_children():
        if name in freeze_modules:
            for p in module.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(FREEZE_MODES.keys()), required=True)
    args = ap.parse_args()
    freeze_modules = FREEZE_MODES[args.mode]

    print(f"Mode: {args.mode}, freezing: {freeze_modules}")
    print("Loading full v8 dataset once...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train_raw = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    s_train = np.load(os.path.join(DATA_DIR, "s_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val_raw = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    s_val = np.load(os.path.join(DATA_DIR, "s_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test_raw = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    s_test = np.load(os.path.join(DATA_DIR, "s_test.npy"))

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    n_classes = len(le.classes_)
    n_channels = X_train.shape[2]

    print("Loading v8 base checkpoint...")
    base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)

    subjects = sorted(np.unique(s_train).tolist())
    per_subject_results = []

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    t_start = time.time()
    for sid in subjects:
        t0 = time.time()
        tr_mask, va_mask, te_mask = s_train == sid, s_val == sid, s_test == sid

        train_ds = EMGDataset(X_train[tr_mask], y_train[tr_mask], s_train[tr_mask])
        val_ds = EMGDataset(X_val[va_mask], y_val[va_mask], s_val[va_mask])
        test_ds = EMGDataset(X_test[te_mask], y_test[te_mask], s_test[te_mask])

        train_loader = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = SubjectAwareCNNLSTMAttention(n_channels=n_channels, n_classes=n_classes, n_subjects=40).to(DEVICE)
        model.load_state_dict(base_ckpt)
        n_trainable, n_total = apply_freeze(model, freeze_modules)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=FT_LR, weight_decay=1e-4)

        _, base_acc, base_f1 = run_epoch(model, test_loader, criterion)

        best_val_f1, best_state, patience_ctr = -1.0, None, 0
        epoch = 0
        for epoch in range(1, FT_EPOCHS + 1):
            run_epoch(model, train_loader, criterion, optimizer)
            _, val_acc, val_f1 = run_epoch(model, val_loader, criterion)
            if val_f1 > best_val_f1:
                best_val_f1, patience_ctr = val_f1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_ctr += 1
                if patience_ctr >= FT_PATIENCE:
                    break

        model.load_state_dict(best_state)
        _, ft_acc, ft_f1 = run_epoch(model, test_loader, criterion)

        elapsed = time.time() - t0
        print(f"[S{sid:02d}] n_train={tr_mask.sum()} trainable_params={n_trainable}/{n_total} "
              f"base_acc={base_acc:.4f} base_f1={base_f1:.4f} -> ft_acc={ft_acc:.4f} ft_f1={ft_f1:.4f} "
              f"(epochs={epoch}, {elapsed:.1f}s)", flush=True)

        per_subject_results.append({
            "subject": int(sid), "n_train": int(tr_mask.sum()),
            "trainable_params": int(n_trainable), "total_params": int(n_total),
            "base_acc": base_acc, "base_f1": base_f1,
            "ft_acc": ft_acc, "ft_f1": ft_f1, "n_epochs": epoch,
        })

    total_time = time.time() - t_start

    base_accs = [r["base_acc"] for r in per_subject_results]
    base_f1s = [r["base_f1"] for r in per_subject_results]
    ft_accs = [r["ft_acc"] for r in per_subject_results]
    ft_f1s = [r["ft_f1"] for r in per_subject_results]

    summary = {
        "mode": args.mode,
        "frozen_modules": freeze_modules,
        "n_subjects": len(subjects),
        "total_time_sec": total_time,
        "pooled_v8_base_mean_acc": float(np.mean(base_accs)),
        "pooled_v8_base_mean_f1": float(np.mean(base_f1s)),
        "finetuned_mean_acc": float(np.mean(ft_accs)),
        "finetuned_mean_f1": float(np.mean(ft_f1s)),
        "finetuned_std_acc": float(np.std(ft_accs)),
        "finetuned_std_f1": float(np.std(ft_f1s)),
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(OUT_DIR, f"finetune_per_subject_{args.mode}.json"), "w") as f:
        json.dump(per_subject_results, f, indent=2)
    with open(os.path.join(OUT_DIR, f"finetune_summary_{args.mode}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
