"""
전처리 파이프라인 v29 — v8 파이프라인 정확 복제

data_utils.py의 실제 순서를 팀 코드 구조에 이식:

  v8 실제 순서 (data_utils.py):
    filter(12ch) → |EMG|(raw) → lowpass env(12ch) → stack 24ch
    → normalize_per_subject(24ch 전체, train_mask=rep∉{5,6})
    → window (rep=0 윈도우 포함)

  v28(잘못된 복제):
    filter → normalize(12ch) → |norm_EMG| → env → stack 24ch

변경 포인트 (v28 대비):
  1. 엔벨로프를 정규화 전 raw filtered EMG에서 계산
  2. 24ch 전체를 jointly 정규화 (12ch EMG + 12ch env)
  3. 정규화 통계에 rep=0 구간 포함 (v8의 train_mask = rep∉{5,6})
  4. 윈도우에서 rep=0 윈도우 포함 (v8과 동일, rep=0 제외 로직 없음)

출력:
  전처리전달_v29/X_{train,val,test}.npy  (N, 600, 24) float32
  전처리전달_v29/y_{train,val,test}.npy  (N,) int64
  전처리전달_v29/s_{train,val,test}.npy  (N,) int64
  전처리전달_v29/norm_mean.npy  (41, 24) zeros  ← 이미 정규화됨
  전처리전달_v29/norm_std.npy   (41, 24) ones
"""
import os, re, gc, glob, shutil, argparse, tempfile
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

# ── 파라미터 (v8과 동일) ─────────────────────────────────────────────────────
FS         = 2000
WIN_MS     = 300
STRIDE_MS  = 150
PURITY     = 0.8
WIN        = int(FS * WIN_MS    / 1000)  # 600
STRIDE     = int(FS * STRIDE_MS / 1000)  # 300

BP_LOW, BP_HIGH = 20.0, 500.0
NOTCH_FREQ      = 50.0
NOTCH_Q         = 30.0
ENV_LOW         = 10.0
ENV_ORDER       = 4

# v8과 동일: rep∉{5,6} = {0,1,2,3,4}가 train_mask
VAL_REPS  = [5]
TEST_REPS = [6]
TRAIN_REPS = [1, 2, 3, 4]  # split용 (rep=0 윈도우는 train에 포함)

EXERCISE_OFFSET = {1: 0, 2: 17, 3: 40}
N_CHANNELS = 24

DATA_DIR = "/mnt/data/medical_project/dataset/ninapro_db2"
OUT_DIR  = "/mnt/data/medical_project/ninapro_db2_project/전처리전달_v29"


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
    out = np.empty_like(emg, dtype=np.float64)
    for c in range(emg.shape[1]):
        x = filtfilt(bp_b, bp_a, emg[:, c])
        out[:, c] = filtfilt(notch_b, notch_a, x)
    return out

def compute_envelope(emg_filt, filters):
    _, _, (env_b, env_a) = filters
    rect = np.abs(emg_filt)
    env  = np.empty_like(rect)
    for c in range(rect.shape[1]):
        env[:, c] = filtfilt(env_b, env_a, rect[:, c])
    return env


# ── 로드 / 라벨 ────────────────────────────────────────────────────────────
def load_mat(path):
    mat      = sio.loadmat(path)
    emg      = mat['emg'].astype(np.float64)
    restim   = mat['restimulus'].flatten().astype(int)
    rerep    = mat['rerepetition'].flatten().astype(int)
    exercise = int(mat['exercise'].flatten()[0])
    n = min(len(emg), len(restim), len(rerep))
    return emg[:n], restim[:n], rerep[:n], exercise

def remap_labels(restim, exercise):
    offset = EXERCISE_OFFSET[exercise]
    out = restim.copy()
    nz  = restim > 0
    out[nz] = restim[nz] + offset
    return out


# ── 윈도우 분할 (rep=0 포함 — v8과 동일) ────────────────────────────────────
def make_windows(feat, labels, reps):
    X, y, r = [], [], []
    T = feat.shape[0]
    for start in range(0, T - WIN + 1, STRIDE):
        end = start + WIN
        wl  = labels[start:end]
        wr  = reps[start:end]

        vals, counts = np.unique(wl, return_counts=True)
        if counts.max() / WIN < PURITY:
            continue
        maj_label = vals[np.argmax(counts)]

        rvals, rcounts = np.unique(wr, return_counts=True)
        maj_rep = rvals[np.argmax(rcounts)]
        # rep=0 윈도우 포함 (v8 동일, segment_signal에 rep=0 제외 없음)

        X.append(feat[start:end])
        y.append(maj_label)
        r.append(maj_rep)

    if not X:
        return (np.empty((0, WIN, N_CHANNELS), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.int64),
            np.array(r, dtype=np.int64))


# ── 피험자 처리 (v8 순서 정확 복제) ──────────────────────────────────────────
def process_subject(subj_dir, filters):
    mats = sorted(glob.glob(os.path.join(subj_dir, '*_E*_A1.mat')))
    if not mats:
        mats = sorted(glob.glob(os.path.join(subj_dir, '*.mat')))

    # 1단계: 모든 exercise 필터링 + 엔벨로프 계산 (정규화 전)
    segments = []
    for mp in mats:
        emg, restim, rerep, exercise = load_mat(mp)
        if exercise not in EXERCISE_OFFSET:
            continue
        emg_f = filter_emg(emg, filters)                # filter
        env   = compute_envelope(emg_f, filters)         # |raw filtered EMG| → lowpass
        feat  = np.concatenate([emg_f, env], axis=1)    # stack 24ch (정규화 전)
        labels = remap_labels(restim, exercise)
        segments.append((feat, labels, rerep))
        del emg, emg_f, env; gc.collect()

    if not segments:
        return None

    # 2단계: 24ch 전체를 jointly 정규화
    # v8의 train_mask = ~isin(reps, {5,6}) = rep∈{0,1,2,3,4} 포함
    train_samples = []
    for feat, _, rerep in segments:
        mask = ~np.isin(rerep, VAL_REPS + TEST_REPS)   # rep∉{5,6}
        if mask.sum() > 0:
            train_samples.append(feat[mask])
    if not train_samples:
        return None

    train_cat = np.concatenate(train_samples, axis=0)  # (N, 24)
    mean = train_cat.mean(axis=0)   # (24,)
    std  = train_cat.std(axis=0) + 1e-8
    del train_cat, train_samples; gc.collect()

    # 3단계: 정규화 적용 후 윈도우 분할
    Xs, ys, rs = [], [], []
    for feat, labels, rerep in segments:
        feat_norm = ((feat - mean) / std).astype(np.float32)
        X, y, r   = make_windows(feat_norm, labels, rerep)
        del feat, feat_norm; gc.collect()
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

    print("v29 전처리: v8 파이프라인 정확 복제")
    print("  순서: filter → env(raw) → stack 24ch → normalize(24ch, rep∉{5,6})")
    print("  rep=0 윈도우: 포함 (v8 동일)")
    print(f"ENV cutoff: {ENV_LOW} Hz  order: {ENV_ORDER}")
    print(f"WIN: {WIN_MS}ms  STRIDE: {STRIDE_MS}ms  PURITY: {PURITY}")
    print(f"출력 경로: {args.out_dir}\n")

    tmpdir   = tempfile.mkdtemp(prefix='db2_v29_', dir=args.out_dir)
    frag_dir = os.path.join(tmpdir, 'frags')
    os.makedirs(frag_dir, exist_ok=True)

    subj_dirs = sorted(glob.glob(os.path.join(args.data_dir, '*')))
    processed = []

    for d in subj_dirs:
        if not os.path.isdir(d):
            continue
        sid  = parse_subject_id(d)
        if sid is None:
            continue
        real = find_mat_dir(d)
        if real is None:
            continue

        print(f'[s{sid:02d}] 처리 중...', end=' ', flush=True)
        res = process_subject(real, filters)
        if res is None:
            print('건너뜀'); continue
        X, y, r = res

        npz = os.path.join(tmpdir, f's{sid}.npz')
        np.savez(npz, X=X, y=y, r=r)
        processed.append((sid, npz))

        tr = np.isin(r, TRAIN_REPS)
        r0 = (r == 0)
        print(f'{len(y):,} 윈도우  '
              f'(train rep1-4:{tr.sum():,} / rep0:{r0.sum():,} / '
              f'val:{np.isin(r,VAL_REPS).sum():,} / test:{np.isin(r,TEST_REPS).sum():,})')
        del X, y, r, res; gc.collect()

    if not processed:
        print('처리된 데이터 없음')
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    processed.sort()
    max_id = max(sid for sid, _ in processed)

    # 분할: train = rep∉{5,6} = {0,1,2,3,4}, val = {5}, test = {6}
    frag_index = {s: [] for s in ['train', 'val', 'test']}
    for sid, npz in processed:
        data = np.load(npz)
        X, y, r = data['X'], data['y'], data['r']
        for split, reps, excl in [
            ('train', None, VAL_REPS + TEST_REPS),   # rep∉{5,6}
            ('val',   VAL_REPS,  None),
            ('test',  TEST_REPS, None),
        ]:
            if excl is not None:
                mask = ~np.isin(r, excl)
            else:
                mask = np.isin(r, reps)
            if mask.sum() == 0:
                continue
            fX = os.path.join(frag_dir, f'{split}_s{sid}_X.npy')
            fy = os.path.join(frag_dir, f'{split}_s{sid}_y.npy')
            fs = os.path.join(frag_dir, f'{split}_s{sid}_s.npy')
            np.save(fX, X[mask])
            np.save(fy, y[mask])
            np.save(fs, np.full(mask.sum(), sid, dtype=np.int64))
            frag_index[split].append((fX, fy, fs))
        del data, X, y, r; gc.collect()

    def merge(split):
        frags = frag_index[split]
        Xpath = os.path.join(args.out_dir, f'X_{split}.npy')
        ypath = os.path.join(args.out_dir, f'y_{split}.npy')
        spath = os.path.join(args.out_dir, f's_{split}.npy')
        if not frags:
            np.save(Xpath, np.empty((0, WIN, N_CHANNELS), dtype=np.float32))
            np.save(ypath, np.empty((0,), dtype=np.int64))
            np.save(spath, np.empty((0,), dtype=np.int64))
            return 0
        counts = [int(np.load(f[1], mmap_mode='r').shape[0]) for f in frags]
        total  = sum(counts)
        Xmm = np.lib.format.open_memmap(Xpath, mode='w+', dtype=np.float32,
                                         shape=(total, WIN, N_CHANNELS))
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

    norm_mean = np.zeros((max_id + 1, N_CHANNELS), dtype=np.float64)
    norm_std  = np.ones( (max_id + 1, N_CHANNELS), dtype=np.float64)
    np.save(os.path.join(args.out_dir, 'norm_mean.npy'), norm_mean)
    np.save(os.path.join(args.out_dir, 'norm_std.npy'),  norm_std)
    shutil.rmtree(tmpdir, ignore_errors=True)

    ids     = [sid for sid, _ in processed]
    missing = [i for i in range(1, 41) if i not in ids]
    print(f'\n완료. 처리 피험자: {len(ids)}/40')
    if missing:
        print(f'누락: {missing}')
    print(f'train: {n_train:,} | val: {n_val:,} | test: {n_test:,}')
    print(f'저장 위치: {args.out_dir}')


if __name__ == '__main__':
    main()
