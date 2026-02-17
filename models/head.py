import torch
import torch.nn as nn


class OrdinalRegressionHead(nn.Module):
    """
    DICAN Reasoning Head - Enhanced Non-Linear 버전
    
    [핵심 변경사항 - 문제4 (Catastrophic Forgetting + CBM 병목) 해결]
    
    1. 입력 차원 확대: 8 → 12 (Multi-Cluster의 Max + Mean + Std)
    2. 비선형 레이어 추가: 기존 3-layer → 4-layer + Residual Connection
       → 더 복잡한 concept 간 상호작용을 학습 (EX*HE 조합 등)
    3. Ordinal-Aware 구조:
       - 첫 번째 블록: Concept 간 교차 학습 (어떤 조합이 어떤 등급인지)
       - Residual 블록: Grade 경계 미세 조정
       - 출력 블록: Ordinal regression (순서 보존)
    
    [기존 문제]
    - 8→32→16→5 (3층)으로는 non-discriminative prototype의 미세한 차이를 포착 불가
    - Prototype 품질이 개선되면 더 깊은 head가 패턴을 학습할 여유가 생김
    
    [구조 설계 근거]
    12 → 64: Concept score 간 교차항(interaction term) 학습
              (예: EX_max + HE_mean → Grade 2 특징)
    64 → 32: Residual block으로 세밀한 boundary 학습
    32 → 16: 압축하면서 노이즈 필터링  
    16 → 5:  최종 등급 예측
    """

    def __init__(self, input_dim=12, num_classes=5, dropout=0.2):
        """
        Args:
            input_dim (int): Concept Score 차원 
                             Multi-Cluster: num_concepts * 3 = 12
            num_classes (int): DR Grade 개수 (0~4 = 5)
            dropout (float): 과적합 방지
        """
        super(OrdinalRegressionHead, self).__init__()
        self.input_dim = input_dim
        
        # =============================================
        # Block 1: Concept Interaction Layer
        # 12개 score 간의 비선형 교차 패턴 학습
        # =============================================
        self.interaction_block = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),  # ReLU보다 smooth → 미세한 score 차이에 민감
            nn.Dropout(dropout),
        )
        
        # =============================================
        # Block 2: Residual Refinement
        # Grade 경계(예: Grade 1 vs 2)를 미세 조정
        # Residual connection으로 gradient vanishing 방지
        # =============================================
        self.residual_block = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.residual_act = nn.GELU()
        
        # =============================================
        # Block 3: Compression + Output
        # =============================================
        self.output_block = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )
        
        self._init_weights()

    def forward(self, concept_scores):
        """
        Args:
            concept_scores: [B, input_dim] 
                           Max(K) + Mean(K) + Std(K) = 3K 차원
        Returns:
            logits: [B, num_classes]
        """
        # Block 1: Interaction
        h = self.interaction_block(concept_scores)  # [B, 64]
        
        # Block 2: Residual
        residual = h
        h = self.residual_block(h)
        h = self.residual_act(h + residual)  # Skip connection
        
        # Block 3: Output
        logits = self.output_block(h)  # [B, 5]
        
        return logits

    def set_session_mode(self, mode):
        if mode == 'base':
            for param in self.parameters():
                param.requires_grad = True
            self.train()
        elif mode == 'incremental':
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# --- 테스트 ---
if __name__ == "__main__":
    head = OrdinalRegressionHead(input_dim=12, num_classes=5)
    
    # Multi-Cluster Score 시뮬레이션
    # [max_EX, max_HE, max_MA, max_SE, mean_EX..., std_EX...]
    severe = torch.tensor([[15.0, 12.0, 8.0, 10.0,  5.0, 4.0, 3.0, 4.0,  2.0, 1.5, 1.0, 1.8]])
    normal = torch.tensor([[-2.0, -3.0, -1.0, -2.0, -1.0, -1.5, -0.5, -1.0,  0.3, 0.2, 0.1, 0.2]])
    
    batch = torch.cat([severe, normal], dim=0)
    
    head.set_session_mode('base')
    logits = head(batch)
    print(f"Logits shape: {logits.shape}")  # [2, 5]
    print(f"Severe prediction: Grade {logits[0].argmax().item()}")
    print(f"Normal prediction: Grade {logits[1].argmax().item()}")
    
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head Parameters: {n_params:,}")
