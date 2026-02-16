import torch
import torch.nn as nn

# 앞서 정의한 모듈들을 상대 경로로 임포트
# (실제 환경에서는 파일 위치에 따라 경로가 달라질 수 있음)
from .backbone import ResNetBackbone
from .projector import ConceptProjector
from .prototypes import PrototypeBank
from .head import OrdinalRegressionHead

class DICAN_CBM(nn.Module):
    """
    DICAN (Domain-Invariant Concept Alignment Network) 전체 모델 클래스.
    
    [구조적 특징]
    - 4개의 핵심 모듈을 조립하여 CBM 파이프라인을 구축함.
    - 'set_session_mode'를 통해 Base Session과 Incremental Session의 
      상이한 동작 방식(Logic)을 하나의 모델 안에서 스위칭함.
    """
    
    def __init__(self, num_concepts=4, num_classes=5, feature_dim=2048):
        super(DICAN_CBM, self).__init__()
        self.mode = 'base' # 기본 모드 설정

        # 1. Backbone (ResNet-50)
        self.backbone = ResNetBackbone(pretrained=True)
        
        # 2. Concept Projector (Learnable Alignment Layer)
        # Base Session에서는 Identity, Inc Session에서는 학습됨.
        self.projector = ConceptProjector(input_dim=feature_dim)
        
        # 3. Prototype Bank (Concept Anchors)
        # 마스크 연산 및 유사도 계산 담당.
        self.prototypes = PrototypeBank(num_concepts=num_concepts, feature_dim=feature_dim)
        
        # 4. Reasoning Head (Ordinal Regression)
        # Concept Score -> DR Grade 예측.
        self.head = OrdinalRegressionHead(num_concepts=num_concepts, num_classes=num_classes)

    def forward(self, x, masks=None):
        """
        Session Mode에 따라 데이터 흐름이 달라지는 Dynamic Forwarding.
        
        Args:
            x (Tensor): Input Image [Batch, 3, 224, 224]
            masks (Tensor, optional): Segmentation Masks [Batch, 4, 224, 224]
                                    (Base Session에서만 필수)
        Returns:
            output (dict): Loss 계산을 위한 모든 중간/최종 결과값 포함
                - 'logits': 최종 DR 등급 예측값 (Ordinal Regression용)
                - 'concept_scores': Concept 유사도/점수 (Intervention용)
                - 'features': (Inc 모드일 때) Projector 통과 후 특징 (Alignment Loss용)
        """
        # 1. Backbone Feature Extraction (공통)
        # [Batch, 2048, 7, 7]
        raw_features = self.backbone(x)
        
        concept_scores = None
        aligned_features = None

        # -----------------------------------------------------------
        # CASE 1: Base Session (Masked GAP -> Prototype Update)
        # -----------------------------------------------------------
        if self.mode == 'base':
            if masks is None:
                raise ValueError("[Base Session] Segmentation masks are required for training!")
            
            # Projector는 사용하지 않음 (Pass-through)
            # 마스크를 사용하여 '진짜 병변' 특징만 추출하고 Prototype 업데이트
            # 리턴값: 현재 배치의 Concept Score (Head 학습용)
            concept_scores = self.prototypes.update_with_masks(raw_features, masks)
            
            # Base Session에서는 Projector 출력이 의미가 없으므로 raw_features를 그대로 둠
            aligned_features = raw_features 

        # -----------------------------------------------------------
        # CASE 2: Incremental Session (Projector -> Similarity)
        # -----------------------------------------------------------
        elif self.mode == 'incremental':
            # Projector를 통과시켜 특징 공간 정렬 (Feature Transformation)
            # [Batch, 2048, 7, 7]
            aligned_features = self.projector(raw_features)
            
            # 저장된 Prototype과의 Cosine Similarity 계산
            # [Batch, num_concepts]
            concept_scores = self.prototypes(aligned_features)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 3. Reasoning Head (Classification) (공통)
        # [Batch, num_classes]
        logits = self.head(concept_scores)
        
        return {
            "logits": logits,
            "concept_scores": concept_scores,
            "features": aligned_features # Alignment Loss 계산을 위해 필요
        }

    def set_session_mode(self, mode):
        """
        모든 서브 모듈에 Session Mode를 전파하여 Freeze/Unfreeze 상태를 동기화함.
        
        Args:
            mode (str): 'base' or 'incremental'
        """
        print(f"[*] Switching DICAN session mode to: {mode.upper()}")
        self.mode = mode
        
        # 각 모듈별로 모드 설정 전파
        self.backbone.set_session_mode(mode)
        self.projector.set_session_mode(mode)
        # PrototypeBank는 nn.Module이지만 학습 파라미터(Weight)가 없으므로 별도 모드 설정 불필요
        # 단, 내부 상태 관리가 필요하다면 추가 가능.
        self.head.set_session_mode(mode)

        # 요약 출력
        if mode == 'base':
            print("   -> Backbone: Trainable (Fine-tuning)")
            print("   -> Projector: Frozen (Identity)")
            print("   -> Prototypes: Updating with Masks")
            print("   -> Head: Trainable")
        elif mode == 'incremental':
            print("   -> Backbone: Frozen")
            print("   -> Projector: Trainable (Learning Alignment)")
            print("   -> Prototypes: Fixed (Reference)")
            print("   -> Head: Frozen")

# --- 테스트 코드 ---
if __name__ == "__main__":
    # 모델 생성
    model = DICAN_CBM()
    
    # 더미 데이터
    images = torch.randn(2, 3, 224, 224)
    masks = torch.randn(2, 4, 224, 224) # Base용 마스크
    
    # 1. Base Session 테스트
    model.set_session_mode('base')
    out_base = model(images, masks)
    print(f"[Base] Logits Shape: {out_base['logits'].shape}")
    print(f"[Base] Concept Scores Shape: {out_base['concept_scores'].shape}")
    
    # 2. Incremental Session 테스트
    model.set_session_mode('incremental')
    # 마스크 없이 이미지(Few-shot)만 입력
    out_inc = model(images) 
    print(f"[Inc] Logits Shape: {out_inc['logits'].shape}")
    # Projector가 작동했는지 확인 (Projector 초기화에 따라 값 변화)
    print(f"[Inc] Features Shape: {out_inc['features'].shape}")