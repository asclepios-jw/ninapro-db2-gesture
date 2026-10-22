# Ninapro DB2 — 팔뚝 근전도로 50가지 손동작 분류

팔뚝 표면 근전도(sEMG) 12채널만으로 **50가지 손동작**을 분류합니다. 40명의 데이터를 **한 모델**에 모아, 사용 중에는 개인 라벨이나 매번의 재보정 없이 쓸 수 있는 범용 모델을 목표로 했습니다. (새 사용자는 처음 한 번 제스처 샘플로 등록하는 과정이 필요합니다. 근전 의수의 캘리브레이션과 같은 방식입니다.)

KBESA(큐베사) 바이오센서팀 프로젝트(2026-06 ~ 08)이고, 저는 Algorithm Leader를 맡았습니다. v1부터 v64까지 64번의 실험을 거쳤습니다.

> **이 레포의 범위**: 최종 모델 아키텍처 코드, 초기 베이스 파이프라인 노트북, 최종 모델(v64) 문서입니다. 학습·앙상블·추론 스크립트(`stability_v57.py`, `inference_v59.py` 등)는 아직 포함하지 않았습니다.

## 결과 (v64, 40명 통합 · 사용 중 개인 라벨 0)

| 평가 단위 | 정확도 | macro-F1 | 어려운 피험자 5명 |
|---|---|---|---|
| 프레임 (300ms 창 하나씩, 실시간) | **73.0%** (95% CI 70.8–75.3) | 0.644 | 63.1% |
| 온라인 평활 (1.5~3초 지연) | 78.6% | | 72.4% |
| 실제 분할 (예측으로 경계 검출) | 80.4% | | 73.4% |
| **동작 단위** (Ninapro 표준, 문헌 비교용) | **88.0%** (95% CI 86.4–89.6) | 0.846 | 83.9% |

- 출발점: 팀 초기 데이터 기준 62.6%(v25) → 73.0%(v48), **+10.4%p**
- seed를 바꿔도 ±0.03%p → 결과 안정
- 한 사람 전용 모델은 82.7%까지 나오지만, 40명을 묶으면 사람마다 신호가 달라 크게 떨어지는 것이 핵심 난점

## 무엇이 성능을 움직였나

| 결정 | 이득 |
|---|---|
| 버그 수정 (손실 보정이 겹쳐 "쉬운 동작을 무시"하던 문제) | +7.5%p |
| 올바른 아키텍처 (초기 급락의 원인은 전처리가 아니라 아키텍처) | +4.7%p |
| 앙상블 (서로 다른 seed·구조 8개 모델의 확률 평균) | +2.7%p |
| 필터 하한 5 → 10 Hz | +1.0%p |
| 라벨을 보존하는 데이터 증강 | +0.8%p |

몇 개의 큰 결정이 대부분을 만들었고, 세밀한 튜닝은 0.1%p 단위였습니다.

## 배운 것

1. **평가 단위가 결론을 바꿉니다.** 프레임 단위 73%, 낮은 macro-F1(0.64)은 "신호의 한계"처럼 보였지만, 실제로는 동작 경계에서 순간적으로 rest로 놓치는 오류였습니다. 동작 단위로 보면 88%, macro-F1 0.85입니다.
2. **용량이 아니라 정보와 다양성입니다.** 파라미터를 1.8배로 늘려도 이득이 없었습니다. 반면 단독으로는 실패한 모델도 앙상블 멤버로는 도움이 됐습니다.
3. **증강은 라벨의 의미를 보존해야 합니다.** 전극 위치를 돌리는 증강은 DB2의 근육 배치와 맞지 않아 −1.6%p였습니다.
4. **강건성은 따로 챙겨야 합니다.** 전극 하나가 고장 나면 −13.5%p였는데, 채널 드롭아웃 증강을 넣은 robust3 모델은 같은 상황에서 70.7%를 유지했습니다(정상 조건에서는 −1.0%p).
5. **외부 검토가 자기 사각지대를 잡습니다.** 다른 모델의 검토로 평가 단위 문제를 발견해 88% 해석으로 이어졌습니다.

## 파이프라인

```
sEMG 12채널 (2 kHz)  →  대역통과 필터 + 포락선 (24채널)  →  300ms 창
   →  1D CNN 계열 모델 (멤버 구조 4종)  →  8모델 앙상블  →  (선택) 시간 후처리
```

최종 모델 구조 사용 예:

```python
import torch
from model.final_model_cnn_lstm_attention import SubjectAwareCNNLSTMAttentionFeat

model = SubjectAwareCNNLSTMAttentionFeat()      # 24채널 입력, 50클래스, 피험자 40명
x = torch.randn(8, 24, 600)                     # (배치, 채널, 300ms @ 2kHz)
subject = torch.randint(1, 41, (8,))            # 피험자 번호 (임베딩 인덱스)
logits = model(x, subject)                      # (8, 50)
```

champion 앙상블 8개 중 3개가 이 구조이고, 나머지는 기반(v8)·주파수 특징(v45)·잔차 CNN(v41) 변형입니다. 배포 구성은 정확도 우선 `champ8`(8모델)과 강건성 우선 `robust3`(3모델, 의수 실사용 가정) 두 가지입니다. 세부 파라미터는 [`docs/final_model_spec_v64.txt`](docs/final_model_spec_v64.txt)에 있습니다.

## 폴더

```
model/final_model_cnn_lstm_attention.py  최종 모델 구조 (v44): CNN + 피험자 임베딩 + BiLSTM + Attention, 시간영역 특징(RMS·MAV·WL·ZC·SSC) 브랜치 결합. 약 82만 파라미터
notebooks/01_baseline_pipeline.ipynb   초기 베이스 파이프라인: .mat 로딩 → 필터·윈도잉 → 특징 → SVM / CNN-1D / CNN-LSTM → 평가 → 실시간 분류기 틀
docs/project_summary_v64.txt           v1~v64 프로젝트 요약 (레버, 막다른 길, 교훈)
docs/final_model_spec_v64.txt          최종 모델 상세 명세 (전처리, 아키텍처, 학습, 앙상블, 성능, 재현 절차)
docs/final_model_code_walkthrough.md   최종 모델 코드를 Python 입문자에게 설명한 글
```

## 데이터
[Ninapro DB2](https://ninapro.hevs.ch/instructions/DB2.html) — 40명, 49개 동작 + rest, Delsys Trigno 12채널 sEMG, 2 kHz. 용량이 커서 레포에 포함하지 않으니 공식 사이트에서 받아 `data/`에 둡니다 (`S1_E1_A1.mat` 형식).

> Atzori M, et al. *Electromyography data for non-invasive naturally-controlled robotic hand prostheses.* Scientific Data 1, 140053 (2014).

## 관련 레포
- [emg_gesture_ai_classifier](https://github.com/asclepios-jw/emg_gesture_ai_classifier) — 이 프로젝트 전, UCI EMG 데이터로 AI 도구 3가지를 비교한 실험
