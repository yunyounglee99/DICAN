import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNetBackbone(nn.Module):
    """
    DICAN 모델을 위한 ResNet-50 Backbone.
    
    [핵심 역할]
    1. Base Session: 이미지의 공간적 특징(Spatial Features)을 추출하여 
      Masked Global Average Pooling(Masked GAP)이 가능하도록 함.
    2. Incremental Session: 파라미터를 Freeze하여 Catastrophic Forgetting을 방지함.
    """

    def __init__(self, pretrained=True, freeze_bn=True):
        """
        Args:
            pretrained (bool): ImageNet Pre-trained 가중치 사용 여부 (Default: True)
            freeze_bn (bool): Base Session 학습 시에도 Batch Norm 통계량을 고정할지 여부.
                              Few-shot이나 작은 배치에서는 True로 두는 것이 안정적일 수 있음.
        """
        super(ResNetBackbone, self).__init__()

        # 1. ResNet-50 로드
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        original_model = models.resnet50(weights=weights)

        # 2. Layer 분리 및 재조립
        # 마지막 FC Layer(classification)와 AvgPool Layer를 제거합니다.
        # 이유: 우리는 단순 분류가 아니라, 7x7 또는 14x14 크기의 
        # Feature Map(공간 정보)이 필요하기 때문입니다 (for Masked GAP).
        self.features = nn.Sequential(*list(original_model.children())[:-2])

        # 3. Feature Dimension 정보 저장 (Projector 등에서 사용)
        self.output_dim = 2048  # ResNet-50의 마지막 채널 수

    def forward(self, x):
        """
        Args:
            x (Tensor): Input Image [Batch, 3, H, W]
        Returns:
            x (Tensor): Feature Map [Batch, 2048, H/32, W/32]
                        (예: 224x224 입력 시 -> 2048x7x7 출력)
        """
        x = self.features(x)
        return x

    def set_session_mode(self, mode):
        """
        학습 단계(Session)에 따라 Backbone의 상태(Frozen/Trainable)를 전환하는 핵심 함수.
        
        Args:
            mode (str): 'base' 또는 'incremental'
        """
        if mode == 'base':
            # Base Session: Backbone 학습 가능 (Fine-tuning)
            # 단, 초기 레이어(Stem 등)는 고정하고 후반부만 푸는 전략도 가능함.
            # 여기서는 전체 Fine-tuning을 기본으로 하되, BN은 선택적으로 제어.
            for param in self.parameters():
                param.requires_grad = True
            self.train() 
            
        elif mode == 'incremental':
            # Incremental Session: Backbone 완전 고정 (Freeze)
            # Replay-Free 환경에서 이전 지식을 잊지 않기 위함.
            for param in self.parameters():
                param.requires_grad = False
            
            # 중요: Eval 모드로 전환하여 Batch Norm 통계량(Running Mean/Var)도 고정해야 함.
            # 이를 안 하면 Few-shot 데이터의 통계량이 전체 분포를 망가뜨림.
            self.eval()
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'base' or 'incremental'.")

    def get_feature_dim(self):
        """다른 모듈(Projector)에서 입력 차원을 알 수 있게 함"""
        return self.output_dim

# --- 테스트 코드 (구현 시 확인용) ---
if __name__ == "__main__":
    # 모델 초기화
    backbone = ResNetBackbone(pretrained=True)
    
    # 1. Base Session 모드 테스트
    backbone.set_session_mode('base')
    dummy_input = torch.randn(2, 3, 224, 224)
    features = backbone(dummy_input)
    print(f"[Base Mode] Output Shape: {features.shape}") 
    # 예상: torch.Size([2, 2048, 7, 7]) -> Masked GAP에 적합한 형태
    
    # 2. Incremental Session 모드 테스트
    backbone.set_session_mode('incremental')
    # 파라미터가 고정되었는지 확인
    print(f"[Inc Mode] Parameter requires_grad: {list(backbone.parameters())[0].requires_grad}")
    # 예상: False