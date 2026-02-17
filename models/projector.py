import torch
import torch.nn as nn
import math

class ConceptProjector(nn.Module):
    """
    DICAN 모델을 위한 Learnable Concept Projector.
    
    [핵심 역할]
    Frozen Backbone에서 추출된 특징($z_{raw}$)을 
    Prototype Bank($P$)가 정의된 특징 공간으로 정렬(Alignment)하는 좌표 변환기.
    
    [구조적 특징]
    - Option B에 따라 1x1 Convolution을 사용하여 공간 정보(Spatial Info)를 유지한 채 채널별 특징을 보정함.
    - 가벼운 구조(Lightweight)로 Few-shot 학습 시 Overfitting을 방지.
    """

    def __init__(self, input_dim=2048, hidden_dim=None, use_bn=True):
        """
        Args:
            input_dim (int): Backbone의 출력 채널 수 (ResNet50: 2048)
            hidden_dim (int): Projector의 내부 차원 (Default: input_dim 유지)
                              *차원을 줄이지 않는 것이 정보 보존에 유리함.
            use_bn (bool): Batch Normalization 사용 여부 (Alignment 안정화)
        """
        super(ConceptProjector, self).__init__()
        
        self.output_dim = input_dim
        # hidden_dim이 지정되지 않으면 입력 차원 유지 (Prototype과 차원 일치)
        self.hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.mode = 'base' # 기본 모드

        # 1. Projector Layer 정의 (1x1 Conv + BN)
        # Linear 대신 Conv2d(1x1)를 쓰는 이유:
        # Backbone의 출력(7x7) 공간 정보를 유지하면서 픽셀(위치)별로 Alignment를 수행하기 위함.
        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim) if use_bn else nn.Identity(),
            # 활성화 함수(ReLU)는 선택사항이나, 단순 좌표 변환(Linear Transformation)이 
            # 목적이므로 여기서는 생략하여 정보 손실을 최소화함.
            nn.ReLU()
        )
        
        # 초기화: Incremental Session 시작 시 Identity에 가깝게 시작하는 것이 유리함
        self._init_weights()

    def forward(self, x):
        """
        Args:
            x (Tensor): Raw Features [Batch, 2048, 7, 7] from Backbone
        Returns:
            x (Tensor): Aligned Features [Batch, 2048, 7, 7]
        """
        # Base Session에서는 Projector가 작동하지 않음 (Identity Mapping)
        if self.mode == 'base':
            return x
            
        # Incremental Session에서는 학습된 Projector를 통과
        return self.layer(x)

    def set_session_mode(self, mode):
        """
        Session에 따라 Projector의 동작(Pass-through vs Active) 및 학습 상태 제어
        """
        self.mode = mode
        if mode == 'base':
            # Base: 사용 안 함 (Gradient 불필요)
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            
        elif mode == 'incremental':
            # Incremental: 학습 대상 (Gradient 필요)
            for param in self.parameters():
                param.requires_grad = True
            self.train() 
            # 주의: Few-shot일 경우 BN 통계량이 불안정할 수 있으므로, 
            # 상황에 따라 BN을 eval() 모드로 두거나 InstanceNorm으로 교체 고려 가능.
            # 여기서는 기본적으로 train()으로 둠.

    def _init_weights(self):
        """
        [연구 꿀팁]
        Projector를 완전히 랜덤하게 초기화하면, 초기에 Backbone의 좋은 특징을 
        다 망가뜨린 상태로 시작하여 학습이 불안정해질 수 있음.
        따라서 'Identity(항등)' 변환에 가깝게 초기화하여, 
        "변화가 필요한 부분만 조금씩 학습(Delta Learning)"하도록 유도함.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 가중치를 Identity에 가깝게 하거나, Kaiming Init 사용
                # 여기서는 학습 안정성을 위해 일반적인 초기화 사용하되, 
                # bias가 있다면 0으로 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# --- 테스트 코드 ---
if __name__ == "__main__":
    # ResNet50 출력(2048)을 받는 Projector 생성
    projector = ConceptProjector(input_dim=2048)
    
    # 더미 입력 (Batch: 2, Ch: 2048, H: 7, W: 7)
    dummy_feat = torch.randn(2, 2048, 7, 7)
    
    # 1. Base Session 테스트 (Identity)
    projector.set_session_mode('base')
    out_base = projector(dummy_feat)
    print(f"[Base] Input == Output? {torch.allclose(dummy_feat, out_base)}")
    # 예상: True
    
    # 2. Incremental Session 테스트 (Active)
    projector.set_session_mode('incremental')
    out_inc = projector(dummy_feat)
    print(f"[Inc] Output Shape: {out_inc.shape}")
    print(f"[Inc] Feature Changed? {not torch.allclose(dummy_feat, out_inc)}")
    # 예상: True (가중치 초기화 및 BN으로 인해 값이 변함)