import torch
import torch.nn as nn

class OrdinalRegressionHead(nn.Module):
    """
    DICAN Reasoning Head - Hybrid Pooling 대응 버전
    
    [변경사항]
    - 입력 차원이 num_concepts(4)에서 num_concepts*2(8)로 확장
    - Max Score(4) + Mean Score(4)를 동시에 받아 진단 논리 학습
    - 의학적 의미: "병변이 있는가(Max)" + "얼마나 퍼졌는가(Mean)" 모두 고려
    
    [Ordinal Regression]
    - DR 등급 순서(0<1<2<3<4)를 보존하는 구조
    - Base Session 이후 Frozen하여 진단 논리 고정
    """

    def __init__(self, input_dim=8, num_classes=5, dropout=0.2):
        """
        Args:
            input_dim (int): Concept Score 차원 (Hybrid: num_concepts * 2 = 8)
            num_classes (int): DR Grade 개수 (0~4 = 5)
            dropout (float): 과적합 방지
        """
        super(OrdinalRegressionHead, self).__init__()
        
        self.input_dim = input_dim

        # [구조 설계 근거]
        # 8 → 32: Concept 간 상호작용 학습 (예: HE_max + EX_mean의 조합)
        # 32 → 16: 압축하면서 노이즈 필터링
        # 16 → 5: 최종 등급 예측
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
        
        self._init_weights()

    def forward(self, concept_scores):
        """
        Args:
            concept_scores: [B, input_dim] 
                           처음 input_dim//2는 Max scores, 나머지는 Mean scores
        Returns:
            logits: [B, num_classes]
        """
        return self.mlp(concept_scores)

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
    head = OrdinalRegressionHead(input_dim=8, num_classes=5)
    
    # Hybrid Score 시뮬레이션
    # [max_EX, max_HE, max_MA, max_SE, mean_EX, mean_HE, mean_MA, mean_SE]
    severe_patient = torch.tensor([[15.0, 12.0, 8.0, 10.0,  5.0, 4.0, 3.0, 4.0]])  # 중증
    normal_patient = torch.tensor([[-2.0, -3.0, -1.0, -2.0, -1.0, -1.5, -0.5, -1.0]])  # 정상
    
    batch = torch.cat([severe_patient, normal_patient], dim=0)
    
    head.set_session_mode('base')
    logits = head(batch)
    print(f"Logits shape: {logits.shape}")  # [2, 5]
    print(f"Severe prediction: Grade {logits[0].argmax().item()}")
    print(f"Normal prediction: Grade {logits[1].argmax().item()}")