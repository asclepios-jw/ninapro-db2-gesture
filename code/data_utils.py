"""
Ninapro DB2 다중 피험자 (S1~S40, E1+E2+E3) 로딩/전처리/세그먼테이션 유틸리티.

기존 notebooks/ninapro_db2_pipeline_claude.ipynb 는 단일 피험자(S1) 프로토타입이며
수정하지 않는다. 이 모듈은 전체 40명 피험자를 풀링하는 새 파이프라인 전용이다.
"""
import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

DATA_DIR = "/mnt/data/medical_project/dataset/ninapro_db2"
FS = 2000
N_CHANNELS = 12
SUBJECTS = list(range(1, 41))
EXERCISES = [1, 2, 3]
LABEL_OFFSETS = {1: 0, 2: 17, 3: 40}  # E1: 0-17, E2: 18-40, E3: 41-49

BP_LOW, BP_HIGH = 20, 500
NOTCH_FREQ = 50
NOTCH_Q = 30
BUTTER_ORDER = 4

WIN_MS, STEP_MS = 200, 100
WIN_LEN = int(WIN_MS / 1000 * FS)
STEP_LEN = int(STEP_MS / 1000 * FS)

TEST_REPS = (6,)
VAL_REPS = (5,)


def load_subject_raw(subject_id: int, data_dir: str = DATA_DIR,
                      exercises: list = EXERCISES) -> dict:
    """피험자 1명의 E1+E2+E3를 이어붙여 로딩 (레이블은 0-49로 오프셋 적용)."""
    all_emg, all_labels, all_reps = [], [], []

    for ex in exercises:
        pattern = os.path.join(data_dir, f"DB2_s{subject_id}", f"DB2_s{subject_id}",
                                f"S{subject_id}_E{ex}_A1.mat")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"missing: {pattern}")

        mat = sio.loadmat(files[0])
        emg = mat["emg"].astype(np.float32)
        labels = mat["restimulus"].astype(np.int32).ravel()
        reps = mat["rerepetition"].astype(np.int32).ravel()

        # 일부 원본 .mat 파일은 emg와 restimulus/rerepetition 길이가 1행씩 어긋남 (데이터 결함)
        n = min(len(emg), len(labels), len(reps))
        emg, labels, reps = emg[:n], labels[:n], reps[:n]

        offset = LABEL_OFFSETS[ex]
        labels_shifted = np.where(labels > 0, labels + offset, 0)

        all_emg.append(emg)
        all_labels.append(labels_shifted)
        all_reps.append(reps)

    return {
        "emg": np.concatenate(all_emg, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "reps": np.concatenate(all_reps, axis=0),
        "subject_id": subject_id,
    }


def bandpass_filter(signal, lowcut=BP_LOW, highcut=BP_HIGH, fs=FS, order=BUTTER_ORDER):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal, freq=NOTCH_FREQ, q=NOTCH_Q, fs=FS):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, signal, axis=0)


def preprocess_emg(emg: np.ndarray) -> np.ndarray:
    emg = bandpass_filter(emg)
    emg = notch_filter(emg)
    return emg.astype(np.float32)


WAVELET = "db4"
WAVELET_LEVEL = 4


def wavelet_denoise(emg: np.ndarray, wavelet: str = WAVELET, level: int = WAVELET_LEVEL) -> np.ndarray:
    """채널별 웨이블릿 소프트 임계값(soft-threshold) 디노이징.
    v12에서 일부 피험자(SNR<10dB)의 신호가 본질적으로 약하다는 게 확인된 뒤,
    bandpass+notch만으로는 못 거른 광대역 잡음을 추가로 줄여 SNR/separability를
    끌어올릴 수 있는지 시험하기 위해 추가."""
    import pywt
    out = np.empty_like(emg)
    for ch in range(emg.shape[1]):
        coeffs = pywt.wavedec(emg[:, ch], wavelet, level=level)
        # 가장 미세한 레벨(detail coeff)의 MAD로 잡음 표준편차 추정 (universal threshold)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(emg)))
        coeffs[1:] = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs, wavelet)
        out[:, ch] = denoised[: emg.shape[0]]
    return out.astype(np.float32)


ENV_LOWPASS = 10
ENV_ORDER = 4


def compute_envelope(emg: np.ndarray, cutoff=ENV_LOWPASS, fs=FS, order=ENV_ORDER) -> np.ndarray:
    """Full-wave rectification + 저역통과 -> RMS-like 진폭 엔벨로프 (채널별)."""
    rectified = np.abs(emg)
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, rectified, axis=0).astype(np.float32)


def normalize_per_subject(emg: np.ndarray, reps: np.ndarray) -> np.ndarray:
    """피험자 자신의 train-rep(1~4) 구간 통계로 채널별 z-score 정규화.
    val/test 통계 유출(leakage) 방지를 위해 train rep만 사용."""
    train_mask = ~np.isin(reps, TEST_REPS + VAL_REPS)
    mean = emg[train_mask].mean(axis=0, keepdims=True)
    std = emg[train_mask].std(axis=0, keepdims=True) + 1e-8
    return ((emg - mean) / std).astype(np.float32)


def apply_label_delay(labels: np.ndarray, reps: np.ndarray, delay_ms: int, fs: int = FS):
    """라벨-EMG 시차 보정. NinaPro의 restimulus는 화면 자극(cue) 시점 기준으로 매겨지는데,
    실제 EMG 반응은 사람의 반응시간(약 100~300ms)만큼 늦게 나타난다. 라벨 타임라인을
    delay_ms만큼 뒤로 밀어서(앞부분은 직전 라벨로 채움) EMG 신호와의 정렬을 맞춘다.
    delay_ms=0이면 기존(v1~v8)과 동일하게 동작한다."""
    if delay_ms <= 0:
        return labels, reps
    delay_len = int(delay_ms / 1000 * fs)
    if delay_len <= 0 or delay_len >= len(labels):
        return labels, reps

    labels_delayed = np.concatenate([np.full(delay_len, labels[0], dtype=labels.dtype), labels[:-delay_len]])
    reps_delayed = np.concatenate([np.full(delay_len, reps[0], dtype=reps.dtype), reps[:-delay_len]])
    return labels_delayed, reps_delayed


def segment_signal(emg: np.ndarray, labels: np.ndarray, reps: np.ndarray,
                    win_len: int = WIN_LEN, step_len: int = STEP_LEN,
                    exclude_rest: bool = False, min_purity: float = 0.0):
    """슬라이딩 윈도우 세그먼테이션. 레이블/반복은 윈도우 내 최빈값.

    min_purity: 윈도우 내 최빈 레이블이 차지하는 비율의 하한선 (0.0 = 필터 없음, v1/v2와 동일).
    동작 전환 경계에 걸친 "오염된" 윈도우(여러 레이블이 섞인 경우)를 제거해 라벨 노이즈를 줄임.
    """
    X_list, y_list, r_list = [], [], []
    n = len(emg)

    for start in range(0, n - win_len + 1, step_len):
        end = start + win_len
        seg_label = labels[start:end]
        seg_rep = reps[start:end]

        counts = np.bincount(seg_label)
        label = counts.argmax()
        purity = counts[label] / win_len
        rep = np.bincount(seg_rep).argmax()

        if exclude_rest and label == 0:
            continue
        if purity < min_purity:
            continue

        X_list.append(emg[start:end])
        y_list.append(label)
        r_list.append(rep)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    r = np.array(r_list, dtype=np.int64)
    return X, y, r


def load_and_segment_subject(subject_id: int, exclude_rest: bool = False,
                              win_ms: int = WIN_MS, step_ms: int = STEP_MS,
                              min_purity: float = 0.0, add_envelope: bool = False,
                              label_delay_ms: int = 0):
    win_len = int(win_ms / 1000 * FS)
    step_len = int(step_ms / 1000 * FS)

    raw = load_subject_raw(subject_id)
    emg = preprocess_emg(raw["emg"])
    raw["labels"], raw["reps"] = apply_label_delay(raw["labels"], raw["reps"], label_delay_ms)

    if add_envelope:
        # rectify+lowpass envelope를 filtered raw 신호 기준으로 계산한 뒤,
        # raw(12ch)와 envelope(12ch)를 합쳐 24ch로 만들고 각각 피험자별 정규화.
        env = compute_envelope(emg)
        combined = np.concatenate([emg, env], axis=1)
        combined = normalize_per_subject(combined, raw["reps"])
    else:
        combined = normalize_per_subject(emg, raw["reps"])

    X, y, r = segment_signal(combined, raw["labels"], raw["reps"], win_len=win_len, step_len=step_len,
                              exclude_rest=exclude_rest, min_purity=min_purity)
    s = np.full(len(y), subject_id, dtype=np.int64)
    return X, y, r, s
