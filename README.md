DICAN 논리 구조

윤영님, 선택하신 "Option B: Feature Transformation (Learnable Projector)" 방식을 기반으로, 교수님께 바로 설명드릴 수 있도록 논문 수준의 구체적인 프레임워크를 정리해 드립니다.
이 구조는 Replay-Free Few-Shot DIL 환경에서 ResNet-50과 Segmentation Data를 활용하여 Prototype-based CBM을 구축하는 가장 이상적인 형태입니다.

[모델 명칭(가칭)]
DICAN: Domain-Invariant Concept Alignment Network
(for Replay-Free Few-Shot Domain Incremental Learning)

1. 전체 모델 아키텍처 (Model Architecture)
   모델은 크게 4가지 모듈로 구성되며, 학습 단계(Session)에 따라 '고정(Freeze)'과 '학습(Update)' 상태가 달라지는 것이 핵심입니다.
1. Backbone ($E$):
   - 구조: ResNet-50 (ImageNet Pre-trained).
   - 역할: 이미지에서 일반적인 시각적 특징(Visual Features) 추출.
   - 상태: Base Session 이후 Frozen (Catastrophic Forgetting 방지).
1. Concept Projector ($\mathcal{T}_\phi$) [핵심]:
   - 구조: 가벼운 Linear Layer (또는 1x1 Conv + BatchNorm).
   - 역할: 새로운 도메인의 특징($z_{raw}$)을 기존 프로토타입 공간에 맞게 좌표 변환(Mapping).
   - 상태: 매 Incremental Session마다 학습 (Trainable).
1. Prototype Bank ($P$):
   - 구조: 각 Concept(출혈, 삼출물 등 $K$개)에 대한 고정된 참조 벡터 집합 $\{p_1, ..., p_K\}$.
   - 역할: "출혈이란 무엇인가"에 대한 변하지 않는 기준점(Anchor).
   - 상태: Base Session에서 생성 후 Fixed.
1. Reasoning Head ($F$):
   - 구조: Ordinal Regression MLP.
   - 역할: Concept 점수를 종합하여 DR 등급($Y$) 판정.
   - 상태: Base Session 이후 Frozen (의학적 진단 논리 보존).

1. Base Session: "Prototype Bank 구축" (The Foundation)
   이 단계의 목표는 분류(Classification)가 아니라, Segmentation Mask를 이용해 **"순도 높은 Concept Prototype"**을 만드는 것입니다.

- 입력 데이터: DDR 데이터셋 (이미지 + 4가지 Lesion Masks + DR 등급).
- 학습 방법 (Masked Global Average Pooling):
  1. Feature Extraction: 이미지를 ResNet-50에 통과시켜 Feature Map $F \in \mathbb{R}^{H \times W \times C}$를 얻습니다.
  2. Mask Alignment: $H \times W$ 크기로 리사이징된 마스크 $M_k$ (Concept $k$에 대한 마스크)를 준비합니다.
  3. Masked Feature Aggregation (구체적 학습법):
     - 일반적인 평균이 아니라, 마스크가 활성화된(병변이 있는) 영역의 픽셀만 골라서 평균을 냅니다.
     - $$p_k^{(i)} = \frac{\sum (F \cdot M_k)}{\sum M_k}$$(이미지 $i$의 $k$번째 Concept 특징)
  4. Bank Construction:
     - 모든 학습 데이터에 대해 위 과정을 수행한 후, 각 Concept 별로 평균 벡터를 계산하여 Prototype Bank에 저장합니다.
     - $$P_k = \text{Average}(p_k^{(i)}) \quad \text{for all } i$$
- Loss Function:
  - $\mathcal{L}_{seg}$: Feature Map이 마스크와 일치하도록 Pixel-level BCE Loss (Backbone 학습용).
  - $\mathcal{L}_{cls}$: 추출된 특징으로 등급 분류 (Reasoning Head 학습용).

3. Incremental Session: "Concept Projector 학습" (The Adaptation)
   새로운 병원 데이터가 Few-shot으로 들어옵니다. 마스크는 없고 등급 라벨만 있습니다. Backbone과 Prototype은 고정하고, Projector($\mathcal{T}_\phi$)만 학습하여 도메인 격차를 줄입니다.

- 입력 데이터: 새로운 병원 이미지 (Few-shot), 등급 라벨 ($Y$). (No Mask, Replay-Free).
- 학습 방법 (Weakly-Supervised Metric Learning):
  1. Feature Extraction: Frozen ResNet-50을 통과하여 $z_{raw}$ 추출. (이때 $z_{raw}$는 도메인 차이로 인해 $P$와 멀리 떨어져 있음).
  2. Projection: 학습할 레이어 $\mathcal{T}_\phi$를 통과시켜 $z_{aligned} = \mathcal{T}_\phi(z_{raw})$ 생성.
  3. Concept Similarity Calculation:
     - $z_{aligned}$와 Prototype Bank $P$ 사이의 Cosine Similarity를 계산하여 Concept Score $S$를 얻습니다.
     - $$S_k = \text{CosSim}(z_{aligned}, P_k)$$
  4. Alignment using Label Logic:
     - 비록 마스크는 없지만, 라벨이 'Severe'라면 "출혈($P_{hemo}$)과 가까워져야 한다"는 논리를 사용하여 Loss를 계산합니다.
- 핵심: 이 과정을 통해 Projector는 **"새로운 병원의 이미지를 옛날 병원의 특징 공간으로 변환하는 법"**을 배웁니다.

4. Loss Function 종류와 구체적 학습 내용
   Incremental Session에서 Projector를 학습시키기 위해 다음 3가지 Loss의 조합을 사용합니다.
   $$\mathcal{L}_{total} = \mathcal{L}_{align} + \lambda_{1}\mathcal{L}_{ordinal} + \lambda_{2}\mathcal{L}_{sparsity}$$

1) Alignment Loss ($\mathcal{L}_{align}$): Metric Learning

- 목적: 라벨($Y$)이 암시하는 Concept과 Feature 간의 거리를 좁힘.
- 학습 내용:
  - 예: 라벨이 $Y \ge 2$ (Moderate 이상)이면, '출혈'과 '삼출물'이 반드시 존재해야 함.
  - 따라서 $z_{aligned}$가 출혈 프로토타입($P_{hemo}$) 및 **삼출물 프로토타입($P_{exudates}$)**과 높은 Cosine Similarity를 갖도록 강제.
  - 수식: Soft Margin Loss 또는 Contrastive Loss 변형 사용.

2. Ordinal Regression Loss ($\mathcal{L}_{ordinal}$): DR 특화

- 목적: DR의 심각도 순서($0 < 1 < 2 < 3 < 4$) 보존 및 Under-diagnosis 방지.
- 학습 내용:
  - 단순 Cross-Entropy 대신, 예측된 등급이 실제 등급보다 낮게 나올 경우(Under-diagnosis) **더 큰 페널티(가중치)**를 부여.
  - Reasoning Head는 고정되어 있지만, 이 Loss는 역전파되어 Projector가 더 명확한 Concept Feature를 만들도록 유도함.

3. Sparsity Constraint ($\mathcal{L}_{sparsity}$): Level 0 불균형 해결

- 목적: 병변이 없는 이미지(Level 0)가 불필요하게 Concept을 활성화하지 않도록 억제.
- 학습 내용:
  - 라벨이 Level 0(Normal)인 경우, 모든 Concept Score($S_k$)의 합이 0에 가까워지도록 L1 Regularization 적용.
  - 이는 Level 0 데이터가 많더라도 "Concept이 없다"는 것을 명확히 학습시켜 False Positive를 줄임.

5. 요약: 교수님 설득 포인트
   이 구조는 윤영님의 연구 목표를 다음과 같이 달성합니다.
1. Replay-Free & Few-Shot: 거대한 모델을 건드리지 않고 가벼운 Projector만 학습하므로, 데이터 저장 없이(Replay-free) 적은 데이터로(Few-shot) 빠르게 적응합니다.
1. Prototype-based CBM: CBM의 구조를 유지하되, Segmentation Mask로 정교하게 깎은 Prototype을 기준점으로 사용하여 설명 가능성(Grounding)을 극대화했습니다.
1. Class Imbalance: Base Session에서 Masked Pooling을 사용했기 때문에, 정상 데이터가 아무리 많아도 프로토타입 생성에 영향을 주지 않아(Noise-free) 희귀 클래스 탐지에 강합니다.

“근데 원래 기존 연구에서도 cnn에서 segmentation 학습시킬때 mask를 압축된 feature map으로 리사이징해서 학습해? 그렇게 하면 정보 손실이 일어나는거 아냐? 내가 잘몰라서 그래” :

윤영님, 박사 과정 연구자로서 정말 날카롭고 정확한 직관을 가지셨습니다. "정보 손실이 일어나지 않느냐?"는 질문은 CNN의 본질을 꿰뚫는 아주 중요한 질문입니다.
결론부터 말씀드리면, 일반적인 Segmentation(U-Net 등)에서는 마스크를 줄이지 않고 Feature를 키우지만(Upsampling), 윤영님의 연구(CBM & Classification)에서는 마스크를 줄여서(Downsampling) Feature Map에 맞추는 것이 더 일반적이고 효율적입니다.
왜 그런지, 그리고 걱정하시는 정보 손실(Information Loss) 문제를 어떻게 바라봐야 하는지 명쾌하게 정리해 드립니다.

1. 두 가지 접근 방식: 목적에 따라 다릅니다
   연구의 목적이 무엇이냐에 따라 "마스크를 줄일지, Feature를 키울지"가 결정됩니다.
   A. 목적이 "정교한 Segmentation"일 때 (예: U-Net, DeepLab)

- 목표: 픽셀 단위로 완벽하게 병변의 경계선을 따는 것.
- 방법: Feature Map을 원본 이미지 크기($224 \times 224$)로 키웁니다(Upsampling/Deconvolution).
- 이유: 마스크를 줄이면 윤영님 말씀대로 경계선 정보가 뭉개져서 디테일이 사라지기 때문입니다.
  B. 목적이 "Feature 추출 및 Classification"일 때 (윤영님의 경우)
- 목표: 병변이 있는 **'영역의 특징(Feature)'**을 뽑아서, "이게 출혈이냐 아니냐"를 판단하는 것.
- 방법: 마스크를 Feature Map 크기($7 \times 7$ 또는 $14 \times 14$)로 줄입니다(Downsampling/Interpolation).
- 이유:
  - CNN의 특성 (Receptive Field): CNN의 마지막 Feature Map의 한 픽셀은 이미 원본 이미지의 약 $32 \times 32$ 픽셀 영역의 정보를 압축해서 담고 있습니다. 즉, Feature Map 자체가 이미 공간 정보를 많이 뭉개놓은 상태입니다.
  - 매칭의 논리: Feature Map 자체가 이미 해상도가 낮기 때문에, 고해상도 마스크를 씌워봤자 의미가 없습니다. "이미 뭉뚱그려진 Feature"에는 "뭉뚱그려진 마스크"를 씌워서 가져오는 것이 차원(Dimension) 상 맞습니다.

2. "정보 손실"에 대한 오해와 진실
   "정보 손실이 일어나는 것 아니냐?" $\rightarrow$ 맞습니다. 공간적(Spatial) 정보 손실은 일어납니다.
   하지만 의미적(Semantic) 정보는 오히려 농축됩니다.

- 손실되는 것: 병변의 "구불구불한 경계선 모양". (어차피 Classification에는 중요하지 않음)
- 보존되는 것: "여기에 병변이 있다"는 위치 정보와 "이것은 출혈이다"라는 특징 정보.
  CBM의 목적은 '출혈의 모양을 그리는 것'이 아니라, **'출혈이라는 개념(Concept)의 값을 계산하는 것'**이므로, 경계선 디테일보다는 **"그 위치에 출혈 특징이 있는가?"**가 훨씬 중요합니다. 따라서 마스크를 줄여서 매칭해도 연구 목표 달성에는 전혀 문제가 없습니다.

3. 윤영님을 위한 "치명적인 팁" (Dr. DR's Tip)
   하지만 윤영님이 다루시는 DR(당뇨망막병증) 데이터셋에는 **Microaneurysms(미세동맥류)**라는 아주 작은 병변이 있습니다.
   이때 마스크를 그냥 Bilinear Interpolation(부드럽게 줄이기)으로 줄이면, 작은 병변이 사라져 버릴(Vanishing) 위험이 있습니다. 이것이 진짜 정보 손실입니다.
   이 문제를 막기 위해 "Max Pooling" 방식의 리사이징을 추천합니다.

- 잘못된 방법 (Bilinear): $32 \times 32$ 영역에 빨간 점 하나가 있으면, 평균을 내버려서 회색이 되어 사라짐.
- 추천 방법 (Max Pooling / Nearest): $32 \times 32$ 영역 안에 단 하나의 픽셀이라도 병변(1)이면, 줄어든 마스크에서도 **"여긴 병변이다(1)"**라고 표시함.
  코드 구현 팁:
  Python

# 마스크 리사이징 할 때 'nearest' 모드나 'max'를 쓰세요.

import torch.nn.functional as F

# mask shape: (Batch, 1, 224, 224) -> (Batch, 1, 7, 7)

resized_mask = F.interpolate(mask, size=(7, 7), mode='nearest')

# 또는 mode='area' (단, 값이 작아질 수 있음)

4. 요약
1. 윤영님 말이 맞습니다. 마스크를 줄이면 디테일 손실이 발생합니다.
1. 하지만 괜찮습니다. 우리 연구(CBM)는 '모양 그리기'가 아니라 '특징 추출'이 목적이기 때문에, CNN Feature Map의 해상도($7 \times 7$)에 맞추는 것이 표준(Standard)입니다.
1. 주의할 점: DR의 미세 병변이 사라지지 않도록, 마스크를 줄일 때 Nearest Neighbor 방식이나 Max Pooling 개념을 적용해서 "작은 점 하나라도 놓치지 않게" 처리한다고 논문에 언급하시면 방어가 완벽합니다.

윤영님, DICAN (Domain-Invariant Concept Alignment Network) 연구를 위해 "Option B" 구조와 학습 파이프라인을 완벽하게 반영한 프로젝트 폴더 구조를 제안해 드립니다.
이 구조는 **Base Session(복잡한 마스크 처리)**과 **Incremental Session(Few-shot & Projector 학습)**이 명확히 분리되도록 설계되었으며, 연구의 재현성과 확장성을 고려한 "박사 과정/학회 제출용" 표준 구조입니다.

📂 DICAN Project Directory Structure
Plaintext

DICAN_Project/
│
├── configs/ # 실험 설정을 관리하는 설정 파일들
│ ├── base_session.yaml # DDR 학습용 (LR, Batch Size, Mask 경로 등)
│ └── incremental_session.yaml # Few-shot DIL용 (Shot 수, Projector LR 등)
│
├── data/ # 데이터 로딩 및 전처리 모듈
│ ├── **init**.py
│ ├── base_loader.py # [Base] 이미지 + Mask 로드 및 전처리
│ ├── incremental_loader.py # [Inc] 이미지 + Label 로드
│ └── sampler.py # [Inc] Few-shot (K-shot N-way) 샘플링 로직
│
├── models/ # 모델 아키텍처 정의 (모듈화가 핵심)
│ ├── **init**.py
│ ├── backbone.py # ResNet-50 (Frozen/Trainable 옵션 포함)
│ ├── projector.py # [핵심] Concept Projector (Learnable Linear Layer)
│ ├── prototypes.py # [핵심] Prototype Bank 관리 및 Similarity 계산 로직
│ ├── head.py # Ordinal Regression MLP (Reasoning Head)
│ └── dican_cbm.py # 위 모듈들을 조립하는 전체 CBM 클래스
│
├── utils/ # 손실 함수 및 유틸리티
│ ├── **init**.py
│ ├── losses.py # [핵심] Alignment, Ordinal, Sparsity, Seg Loss 구현
│ ├── metrics.py # Accuracy, Kappa Score, Concept RMSE 계산
│ └── visualization.py # 결과 시각화 (Confusion Matrix, Saliency Map)
│
├── saved_prototypes/ # [중요] Base Session에서 추출한 Prototype 벡터 저장소
│ └── ddr_prototypes.pt # Tensor 형태로 저장된 Anchor Vectors
│
├── checkpoints/ # 모델 가중치 저장 (.pth)
│ ├── base_session/ # Base Session 완료된 모델
│ └── incremental/ # 각 에피소드(병원)별 Projector 가중치
│
├── train_base.py # [Phase 1] Base Session 실행 스크립트 (Masked GAP)
├── train_incremental.py # [Phase 2] Incremental Session 실행 스크립트 (Projector Tuning)
├── eval.py # 평가 스크립트 (Forgetting, Adaptation 측정)
│
└── requirements.txt # 필요한 라이브러리 목록

🛠️ 주요 파일별 역할 및 구현 가이드
연구의 핵심 로직이 어디에 들어가야 하는지 구체적으로 매핑해 드립니다.

1. models/dican_cbm.py (전체 조립)
   교수님께 설명드린 4가지 모듈을 여기서 조립합니다.

- 역할: Backbone, Projector, PrototypeBank, ReasoningHead를 인스턴스화하고 연결합니다.
- 핵심 로직: Session 모드(base vs incremental)에 따라 requires_grad를 켜고 끄는(Freezing) 로직이 **init**이나 set_mode() 함수에 포함되어야 합니다.

2. models/prototypes.py (Prototype Bank)

- 역할: 프로토타입을 계산하고 저장하며, 코사인 유사도를 계산합니다.
- 포함될 함수:
  - update_with_mask(features, masks): Base Session에서 Masked GAP를 수행하여 프로토타입을 업데이트하는 함수.
  - compute_similarity(aligned_features): Projector를 통과한 특징과 저장된 프로토타입 간의 유사도(Concept Score)를 반환하는 함수.

3. models/projector.py (Option B의 핵심)

- 구조: 아주 간단하지만 가장 중요한 파일입니다.
- 내용: nn.Linear 혹은 nn.Sequential(nn.Conv2d(1x1), nn.BatchNorm) 정의.
- 특이사항: 매 Incremental Session이 시작될 때마다 초기화(Re-init)하거나 이전 가중치를 불러오는 로직을 train_incremental.py에서 제어합니다.

4. utils/losses.py (Loss Function)
   논문에서 정의한 3가지 Loss를 수식 그대로 구현합니다.

- AlignmentLoss: 라벨 논리에 따른 Cosine Embedding Loss.
- OrdinalLoss: 예측 등급과 실제 등급의 차이에 가중치를 둔 Cross Entropy 변형.
- SparsityLoss: L1 Regularization.

5. train_base.py (Phase 1)

- 데이터: base_loader.py 사용 (이미지 + 마스크).
- 학습 대상: Backbone, Reasoning Head. (Projector는 Identity).
- 출력: saved_prototypes/ddr_prototypes.pt (이 파일이 Phase 2의 씨앗이 됩니다).

6. train_incremental.py (Phase 2)

- 데이터: incremental_loader.py 사용 (이미지 + 라벨). Mask 사용 안 함.
- 로딩: checkpoints/base_session/model.pth와 saved_prototypes/ddr_prototypes.pt를 로드.
- Freezing: Backbone, Head, Prototype 고정.
- 학습 대상: 오직 models/projector.py만 학습.
- 알고리즘: Few-shot 샘플링 -> Projector 통과 -> Similarity 계산 -> Loss 역전파 -> Projector 업데이트.

💡 연구 팁: 실험 관리
이 구조대로 코드를 짜시면, 나중에 "Ablation Study" (예: Projector 유무에 따른 성능 차이, Loss 조합에 따른 변화 등)를 할 때 매우 편합니다. configs/ 폴더의 yaml 파일만 수정하면 되기 때문입니다.
이제 이 폴더 구조를 바탕으로 하나씩 채워 넣으시면, 교수님께 보여드릴 때도 **"체계적으로 설계된 연구"**라는 인상을 강하게 줄 수 있습니다.
