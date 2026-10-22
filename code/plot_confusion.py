"""저장된 confusion matrix(.npy)를 정규화 히트맵으로 시각화."""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v2"


def plot(model_name: str):
    cm_path = os.path.join(OUT_DIR, f"{model_name}_confusion_matrix.npy")
    cm = np.load(cm_path)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                cbar_kws={"label": "Recall"})
    ax.set_title(f"Confusion Matrix (Normalized) — {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{model_name}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnnlstm")
    args = ap.parse_args()
    plot(args.model)
