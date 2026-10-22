"""
현재 EMG 제스처 분류 모델을 LLM(MiniMax-M3 via NVIDIA NIM)에게 평가받는 스크립트.

사용법:
  export NVIDIA_API_KEY="nvapi-xxxx"
  python evaluate_with_llm.py
"""
import os
import sys
import json
import requests

API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
URL      = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL    = "minimaxai/minimax-m3"
OUT_PATH = "/mnt/data/medical_project/ninapro_db2_project/results/llm_evaluation.txt"

if not API_KEY:
    sys.exit("[오류] 환경변수 NVIDIA_API_KEY가 설정되지 않았습니다.\n"
             "  export NVIDIA_API_KEY='nvapi-xxxx'")

SYSTEM_PROMPT = (
    "You are an expert in deep learning for biosignal processing, specifically "
    "surface EMG gesture recognition. Provide precise, actionable feedback on "
    "model architecture, training strategy, and experimental design. Be direct and technical."
)

USER_CONTENT = """## Project: sEMG Gesture Classification (Ninapro DB2)

Dataset: 40 subjects, 50 classes, 12-ch sEMG at 2kHz + 12 envelope = 24 channels total.
Split: reps 1-4 train / rep 5 val / rep 6 test (rep-based, NOT random).
Window: 300ms, stride 150ms, purity 0.8. Input shape: (N, 600, 24).
Per-subject train windows: ~11,190.

## Best Model Architecture ("SubjectAwareCNNLSTMAttention")

Input (B, 24, 600)
→ Conv1d(24→64, k=3) + BN + ReLU
→ Conv1d(64→128, k=3) + BN + ReLU        # (B, 128, 600)
→ BiLSTM(128, hidden=128, layers=2, dropout=0.3)  # (B, 600, 256)
→ Attention weighted sum                  # (B, 256)
→ Concat with SubjectEmbedding(40, 32)   # (B, 288)
→ Dropout(0.4) + Linear(288, 50)

Training: AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR, label_smoothing=0.1, batch=256, RTX 2070 8GB.

## Experiment Results (v8–v21)

| Ver  | Description                          | Overall | Hard(5sub) | Easy(35sub) | Gap   |
|------|--------------------------------------|---------|------------|-------------|-------|
| v8   | Pooled, 30ep                         | 68.69%  | 57.52%     | 70.29%      | 12.77p|
| v10  | Pooled, 50ep (current best)          | 69.10%  | ~58%       | ~71%        | ~13p  |
| v18  | DANN 40-class domain, no subj emb    | 59.10%  | 47.47%     | 60.75%      | 13.28p|
| v19  | DANN binary hard/easy, subj emb kept | 62.27%  | 68.33%     | 61.42%      | -6.90p|
| v20  | DANN v19, lambda 1.0→0.1             | 63.00%  | 69.01%     | 62.16%      | -6.85p|
| v21  | 40 independent per-subject models    | 69.00%  | 57.59%     | 70.63%      | 13.04p|

Hard subjects: {4, 7, 26, 28, 31} = 12.42% of data.

## Key Findings

1. Pooling cost is negligible: individual models (v21 69.00%) ≈ pooled (v8 68.69%). 40x more data, same accuracy.
2. Hard/easy gap is intrinsic to the data: even dedicated per-subject models show ~13p gap. Not caused by cross-subject interference.
3. DANN failed: binary hard/easy domain with 12%/88% imbalance caused GRL to push features hard-like, hurting easy subjects. Lambda tuning (1.0→0.1) had no directional effect.
4. Architecture plateau at ~69%: unchanged CNN+BiLSTM+attention since v8, all approaches converge here.

## Questions

1. Architecture critique: Is CNN+BiLSTM+attention appropriate for 300ms sEMG at 2kHz, 50-class, 40 subjects? What are the bottlenecks at 69%?
2. Hard subject strategy: hard subjects plateau at ~58% even with dedicated models. What signal-level, architecture-level, or training-level approaches would you recommend?
3. Breaking 69% plateau: what single change (architecture, training, data) is most likely to push above 72%?
4. DANN critique: was binary hard/easy domain adaptation fundamentally flawed here? How would you redesign it?
5. Priority: given RTX 2070 8GB and limited experiment budget, which experiment do you recommend next?
"""


def main():
    print(f"모델: {MODEL}")
    print("=" * 60)
    print("전송 중...\n")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_CONTENT},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False,
    }

    resp = requests.post(URL, headers=headers, json=payload, timeout=180)

    if resp.status_code != 200:
        print(f"[오류] HTTP {resp.status_code}")
        print(resp.text[:500])
        sys.exit(1)

    data   = resp.json()
    answer = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", "N/A")

    print(answer)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL}\n{'='*60}\n\n")
        f.write("[PROMPT]\n" + USER_CONTENT + "\n\n")
        f.write(f"{'='*60}\n\n[LLM RESPONSE]\n" + answer + "\n")

    print(f"\n{'='*60}")
    print(f"결과 저장: {OUT_PATH}")
    print(f"사용 토큰: {tokens}")


if __name__ == "__main__":
    main()
