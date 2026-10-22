"""rf_result.json / cnn1d_result.json / cnnlstm_result.json 모아서 비교표 출력."""
import json
import os

OUT_DIR = "/mnt/data/medical_project/ninapro_db2_project/results/v2"
FILES = ["rf_result.json", "cnn1d_result.json", "cnnlstm_result.json"]

rows = []
for fn in FILES:
    path = os.path.join(OUT_DIR, fn)
    if os.path.exists(path):
        with open(path) as f:
            rows.append(json.load(f))

if not rows:
    print("No results found yet.")
else:
    print(f"{'Model':<14}{'Test Acc':>10}{'Test MacroF1':>14}{'Train Time(s)':>16}")
    for r in rows:
        print(f"{r['model']:<14}{r['test_accuracy']:>10.4f}{r['test_macro_f1']:>14.4f}{r['train_time_sec']:>16.1f}")
