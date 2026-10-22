"""
전처리 파이프라인 (팀 코드 기반, ENV 10Hz/4차 수정본)

팀 원본 대비 변경:
  ENV_LOW : 5.0 Hz → 10.0 Hz
  env 필터 차수: 2차  → 4차

그 외 모든 파라미터(윈도우, stride, purity, 분할, 정규화)는 팀 스펙 그대로.

출력:
  OUT_DIR/X_train.npy  (N, 600, 24)  float32
  OUT_DIR/X_val.npy
  OUT_DIR/X_test.npy
  OUT_DIR/y_train.npy  (N,)  int64   라벨 0~49
  OUT_DIR/y_val.npy
  OUT_DIR/y_test.npy
  OUT_DIR/s_train.npy  (N,)  int64   피험자 ID 1~40
  OUT_DIR/s_val.npy
  OUT_DIR/s_test.npy
  OUT_DIR/norm_mean.npy  (41, 24)  float64
  OUT_DIR/norm_std.npy   (41, 24)  float64
"""
import os
import re
import gc
import glob
import shutil
import argparse
import tempfile
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

# ── 파라미터 ────────────────────────────────────────────────────────────────
FS         = 2000
WIN_MS     = 300
STRIDE_MS  = 150
PURITY     = 0.8
WIN        = int(FS * WIN_MS    / 1000)   # 600
STRIDE     = int(FS * STRIDE_MS / 1000)   # 300

BP_LOW,  BP_HIGH = 20.0, 500.0
NOTCH_FREQ       = 50.0
NOTCH_Q          = 30.0

ENV_LOW   = 10.0   # ← 팀 5Hz → 10Hz 수정
ENV_ORDER = 4      # ← 팀 2차  → 4차  수정

TRAIN_REPS = [1, 2, 3, 4]
VAL_REPS   = [5]
TEST_REPS  = [6]

EXERCISE_OFFSET = {1: 0, 2: 17, 3: 40}
N_CHANNELS = 24   # 12 raw EMG + 12 envelope

DATA_DIR = "/mnt/data/medical_project/dataset/ninapro_db2"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_10hz"


# ── 유틸 ────────────────────────────────────────────────────────────────────
def parse_subject_id(path):
    name = os.path.basename(path.rstrip('/\\'))
    m = re.search(r'[sS](\d+)', name)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else None


def find_mat_dir(path):
    if glob.glob(os.path.join(path, '*.mat')):
        return path
    for sub in sorted(glob.glob(os.path.join(path, '*'))):
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, '*.mat')):
            return sub
    return None


# ── 필터 ────────────────────────────────────────────────────────────────────
def make_filters():
    nyq = FS / 2.0
    bp_b, bp_a       = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype='band')
    notch_b, notch_a = iirnotch(NOTCH_FREQ, NOTCH_Q, FS)
    env_b, env_a     = butter(ENV_ORDER, ENV_LOW / nyq, btype='low')
    return (bp_b, bp_a), (notch_b, notch_a), (env_b, env_a)


def filter_emg(emg, filters):
    (bp_b, bp_a), (notch_b, notch_a), _ = filters
    out = np.empty_like(emg, dtype=np.float32)
    for c in range(emg.shape[1]):
        x = filtfilt(bp_b, bp_a, emg[:, c])
        x = filtfilt(notch_b, notch_a, x)
        out[:, c] = x.astype(np.float32)
    return out


def envelope(emg_filt, filters):
    _, _, (env_b, env_a) = filters
    rect = np.abs(emg_filt)
    env  = np.empty_like(rect, dtype=np.float32)
    for c in range(rect.shape[1]):
        env[:, c] = filtfilt(env_b, env_a, rect[:, c]).astype(np.float32)
    return env


# ── 로드 / 라벨 ────────────────────────────────────────────────────────────
def load_mat(path):
    mat     = sio.loadmat(path)
    emg     = mat['emg'].astype(np.float32)
    restim  = mat['restimulus'].flatten().astype(int)
    rerep   = mat['rerepetition'].flatten().astype(int)
    exercise = int(mat['exercise'].flatten()[0])
    n = min(len(emg), len(restim), len(rerep))
    return emg[:n], restim[:n], rerep[:n], exercise


def remap_labels(restim, exercise):
    offset = EXERCISE_OFFSET[exercise]
    out    = restim.copy()
    nz     = restim > 0
    out[nz] = restim[nz] + offset
    return out


# ── 윈도우 분할 ─────────────────────────────────────────────────────────────
def make_windows(feat, labels, reps):
    X, y, r = [], [], []
    T = feat.shape[0]
    for start in range(0, T - WIN + 1, STRIDE):
        end  = start + WIN
        wl   = labels[start:end]
        wr   = reps[start:end]

        vals, counts = np.unique(wl, return_counts=True)
        if counts.max() / WIN < PURITY:
            continue
        maj = vals[np.argmax(counts)]

        rvals, rcounts = np.unique(wr, return_counts=True)
        maj_rep = rvals[np.argmax(rcounts)]
        if maj_rep == 0:          # rep=0 전이 구간 폐기
            continue

        X.append(feat[start:end])
        y.append(maj)
        r.append(maj_rep)

    if not X:
        return (np.empty((0, WIN, feat.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.int64),
            np.array(r, dtype=np.int64))


# ── 피험자 처리 ─────────────────────────────────────────────────────────────
def process_subject(subj_dir, filters):
    mats = sorted(glob.glob(os.path.join(subj_dir, '*_E*_A1.mat')))
    if not mats:
        mats = sorted(glob.glob(os.path.join(subj_dir, '*.mat')))

    Xs, ys, rs = [], [], []
    for mp in mats:
        emg, restim, rerep, exercise = load_mat(mp)
        if exercise not in EXERCISE_OFFSET:
            continue
        labels = remap_labels(restim, exercise)
        emg_f  = filter_emg(emg, filters)
        feat   = np.concatenate([emg_f, envelope(emg_f, filters)], axis=1)
        X, y, r = make_windows(feat, labels, rerep)
        del emg, emg_f, feat
        gc.collect()
        if len(X):
            Xs.append(X); ys.append(y); rs.append(r)

    if not Xs:
        return None
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    r = np.concatenate(rs)
    del Xs, ys, rs; gc.collect()
    return X, y, r


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--out_dir',  default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    filters = make_filters()

    print(f"ENV cutoff: {ENV_LOW} Hz  order: {ENV_ORDER}")
    print(f"WIN: {WIN_MS}ms  STRIDE: {STRIDE_MS}ms  PURITY: {PURITY}")
    print(f"출력 경로: {args.out_dir}\n")

    tmpdir   = tempfile.mkdtemp(prefix='db2_tmp_', dir=args.out_dir)
    frag_dir = os.path.join(tmpdir, 'frags')
    os.makedirs(frag_dir, exist_ok=True)

    subj_dirs = sorted(glob.glob(os.path.join(args.data_dir, '*')))
    processed = []
    sum_, sumsq_, cnt_ = {}, {}, {}

    for d in subj_dirs:
        if not os.path.isdir(d):
            continue
        sid = parse_subject_id(d)
        if sid is None:
            continue
        real = find_mat_dir(d)
        if real is None:
            continue

        print(f'[s{sid:02d}] 처리 중...', end=' ', flush=True)
        res = process_subject(real, filters)
        if res is None:
            print('건너뜀 (0 윈도우)')
            continue
        X, y, r = res

        # 정규화 통계: train rep 윈도우만 사용
        tr = np.isin(r, TRAIN_REPS)
        if tr.sum() > 0:
            flat       = X[tr].reshape(-1, N_CHANNELS).astype(np.float64)
            sum_[sid]  = flat.sum(0)
            sumsq_[sid] = (flat ** 2).sum(0)
            cnt_[sid]  = flat.shape[0]
            del flat

        npz = os.path.join(tmpdir, f's{sid}.npz')
        np.savez(npz, X=X, y=y, r=r)
        processed.append((sid, npz))
        print(f'{len(y):,} 윈도우  (train:{tr.sum():,} / val:{np.isin(r,VAL_REPS).sum():,} / test:{np.isin(r,TEST_REPS).sum():,})')
        del X, y, r, res; gc.collect()

    if not processed:
        print('처리된 데이터 없음')
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    processed.sort()
    max_id = max(sid for sid, _ in processed)

    # 피험자별 정규화 통계 계산
    norm_mean = np.zeros((max_id + 1, N_CHANNELS), dtype=np.float64)
    norm_std  = np.ones( (max_id + 1, N_CHANNELS), dtype=np.float64)
    for sid in cnt_:
        n    = cnt_[sid]
        mean = sum_[sid] / n
        var  = sumsq_[sid] / n - mean ** 2
        std  = np.sqrt(np.maximum(var, 1e-12)) + 1e-8
        norm_mean[sid] = mean
        norm_std[sid]  = std

    # 정규화 적용 후 분할별 저장 (fragment)
    frag_index = {s: [] for s in ['train', 'val', 'test']}
    for sid, npz in processed:
        data = np.load(npz)
        X, y, r = data['X'], data['y'], data['r']
        mean = norm_mean[sid].astype(np.float32)
        std  = norm_std[sid].astype(np.float32)
        Xn   = ((X - mean) / std).astype(np.float32)

        for split, reps in [('train', TRAIN_REPS), ('val', VAL_REPS), ('test', TEST_REPS)]:
            mask = np.isin(r, reps)
            if mask.sum() == 0:
                continue
            fX = os.path.join(frag_dir, f'{split}_s{sid}_X.npy')
            fy = os.path.join(frag_dir, f'{split}_s{sid}_y.npy')
            fs = os.path.join(frag_dir, f'{split}_s{sid}_s.npy')
            np.save(fX, Xn[mask])
            np.save(fy, y[mask])
            np.save(fs, np.full(mask.sum(), sid, dtype=np.int64))
            frag_index[split].append((fX, fy, fs))
        del data, X, y, r, Xn; gc.collect()

    # 분할별 병합
    def merge(split):
        frags  = frag_index[split]
        Xpath  = os.path.join(args.out_dir, f'X_{split}.npy')
        ypath  = os.path.join(args.out_dir, f'y_{split}.npy')
        spath  = os.path.join(args.out_dir, f's_{split}.npy')
        if not frags:
            np.save(Xpath, np.empty((0, WIN, N_CHANNELS), dtype=np.float32))
            np.save(ypath, np.empty((0,), dtype=np.int64))
            np.save(spath, np.empty((0,), dtype=np.int64))
            return 0
        counts = [int(np.load(f[1], mmap_mode='r').shape[0]) for f in frags]
        total  = sum(counts)
        Xmm = np.lib.format.open_memmap(Xpath, mode='w+', dtype=np.float32, shape=(total, WIN, N_CHANNELS))
        ymm = np.lib.format.open_memmap(ypath, mode='w+', dtype=np.int64,   shape=(total,))
        smm = np.lib.format.open_memmap(spath, mode='w+', dtype=np.int64,   shape=(total,))
        pos = 0
        for f, c in zip(frags, counts):
            Xmm[pos:pos+c] = np.load(f[0])
            ymm[pos:pos+c] = np.load(f[1])
            smm[pos:pos+c] = np.load(f[2])
            pos += c
            Xmm.flush(); ymm.flush(); smm.flush()
            gc.collect()
        del Xmm, ymm, smm; gc.collect()
        return total

    print('\n[병합 중...]')
    n_train = merge('train')
    n_val   = merge('val')
    n_test  = merge('test')

    np.save(os.path.join(args.out_dir, 'norm_mean.npy'), norm_mean)
    np.save(os.path.join(args.out_dir, 'norm_std.npy'),  norm_std)
    shutil.rmtree(tmpdir, ignore_errors=True)

    ids     = [sid for sid, _ in processed]
    missing = [i for i in range(1, 41) if i not in ids]
    print(f'\n완료. 처리 피험자: {len(ids)}/40')
    if missing:
        print(f'누락: {missing}')
    print(f'train: {n_train:,} | val: {n_val:,} | test: {n_test:,}')
    print(f'윈도우 shape: ({WIN}, {N_CHANNELS})')
    print(f'저장 위치: {args.out_dir}')


if __name__ == '__main__':
    main()
