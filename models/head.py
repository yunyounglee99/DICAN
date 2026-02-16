import torch
import torch.nn as nn

class OrdinalRegressionHead(nn.Module):
    """
    DICAN 모델의 추론 모듈 (Reasoning Head).
    
    [핵심 역할]
    1. Input: PrototypeBank에서 계산된 'Concept Score' (예: [출혈점수, 삼출물점수...])
    2. Output: DR 등급 (Level 0 ~ 4)에 대한 Logits.
    3. Logic: "출혈 점수가 높으면 중증이다"와 같은 의학적 인과관계를 학습함.
    
    [Ordinal Regression 구조]
    - DR 등급은 순서가 중요하므로(0 < 1 < 2 < 3 < 4), 단순 분류보다는 
      순서 정보를 보존할 수 있는 구조가 유리함.
    - Base Session 이후에는 이 논리가 변하면 안 되므로 파라미터를 Freeze함.
    """

    def __init__(self, num_concepts=4, num_classes=5, num_sub_concepts=5, hidden_dims=[32, 16], dropout=0.0):
        """
        Args:
            num_concepts (int): 입력 차원 (Concept 개수, DDR=4)
            num_classes (int): 출력 클래스 개수 (DR Grade=5)
            hidden_dims (list): MLP 은닉층의 차원 리스트
            dropout (float): 과적합 방지를 위한 Dropout 비율
        """
        super(OrdinalRegressionHead, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # 1. MLP 레이어 구축
        layers = []
        in_dim = num_concepts * num_sub_concepts
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # 입력값(Concept Score) 분포 안정화
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        # 2. 최종 출력층 (Logits for Ordinal Loss)
        # 중요: Ordinal Regression에서는 보통 (Class - 1)개의 이진 분류기를 두거나
        # 단순히 Class 개수만큼의 Logits를 뽑고 Loss에서 처리하기도 함.
        # 여기서는 가장 유연한 방식인 'Class 개수만큼 Logits 출력'을 사용.
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._init_weights()

    def forward(self, concept_scores):
        """
        Args:
            concept_scores (Tensor): [Batch, num_concepts] (0~1 사이의 유사도 값)
        Returns:
            logits (Tensor): [Batch, num_classes] (Softmax 전의 값)
        """
        return self.mlp(concept_scores)

    def set_session_mode(self, mode):
        """
        학습 단계(Session)에 따라 Head의 상태를 제어하는 함수.
        
        [전략]
        - Base Session: 학습 가능 (진단 논리를 배움)
        - Incremental Session: 완전 고정 (Freeze)
          -> 병원이 바뀌어도 "출혈이 많으면 중증"이라는 사실은 변하지 않기 때문.
          -> Head가 고정되어야 Projector가 Head의 기준에 맞춰 Feature를 정렬(Align)하려고 노력함.
        """
        if mode == 'base':
            # Base: 학습 모드
            for param in self.parameters():
                param.requires_grad = True
            self.train()
            
        elif mode == 'incremental':
            # Incremental: 고정 모드 (Frozen Logic)
            for param in self.parameters():
                param.requires_grad = False
            self.eval() # BN 통계량도 고정
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_weights(self):
        """학습 초기 안정성을 위한 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# --- 테스트 코드 ---
if __name__ == "__main__":
    # Head 초기화 (Concept 4개 -> Class 5개)
    head = OrdinalRegressionHead(num_concepts=4, num_classes=5)
    
    # 더미 입력 (Batch=2, Concept=4)
    # 예: 첫 번째 환자는 Concept 점수가 높음 (중증), 두 번째는 낮음 (정상)
    dummy_scores = torch.tensor([[0.9, 0.8, 0.7, 0.9], 
                                [0.1, 0.0, 0.1, 0.0]])
    
    # 1. Base Session 테스트
    head.set_session_mode('base')
    logits = head(dummy_scores)
    print(f"[Base] Logits Shape: {logits.shape}") # [2, 5]
    
    # 2. Incremental Session 테스트
    head.set_session_mode('incremental')
    print(f"[Inc] Requires Grad: {list(head.parameters())[0].requires_grad}") # False