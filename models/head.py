import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionHead(nn.Module):
    """
    DICAN Reasoning Head v2 - Enhanced Non-Linear Architecture
    
    [기존 문제점]
    8 → 32 → 16 → 5 (3 layers, ~1.4K params)
    - 너무 얕아서 concept score 간 복잡한 상호작용 학습 불가
    - Grade 3 vs 4 구분처럼 미묘한 차이를 포착 못함
    - BatchNorm이 small batch에서 불안정
    
    [v2 개선사항]
    1. Feature Interaction Layer:
       - 8개 원본 score + 16개 pairwise interaction = 24차원 입력
       - "EX_max × HE_mean" 같은 cross-concept 상호작용 포착
    
    2. Deeper Architecture with Residual:
       24 → 128 → 128 (residual) → 64 → 32 → 5
       - 충분한 capacity로 비선형 매핑 학습
       - Residual connection으로 gradient vanishing 방지
    
    3. LayerNorm + GELU:
       - LayerNorm: batch size 무관하게 안정적 (Few-shot 호환)
       - GELU: ReLU보다 부드러운 활성화 → 미세한 차이 학습에 유리
    
    4. Ordinal Bias Initialization:
       - 최종 layer bias를 ordinal 순서에 맞게 초기화
       - 학습 초기부터 "등급 순서" 반영
    """

    def __init__(self, input_dim=8, num_classes=5, dropout=0.15):
        super(OrdinalRegressionHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_concepts = input_dim // 2  # Max + Mean → concept 수
        
        # ─── 1. Feature Interaction Layer ───
        # 원본 8개 + pairwise products (max_i * mean_j) 16개 = 24개
        self.interaction_dim = input_dim + self.num_concepts * self.num_concepts
        
        # ─── 2. Input Projection ───
        self.input_proj = nn.Sequential(
            nn.Linear(self.interaction_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ─── 3. Residual Block ───
        self.res_block = ResidualBlock(128, dropout=dropout)
        
        # ─── 4. Compression + Classification ───
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )
        
        self._init_weights()
        self._init_ordinal_bias(num_classes)

    def forward(self, concept_scores):
        """
        Args:
            concept_scores: [B, input_dim]
                           처음 num_concepts개 = Max scores
                           나머지 num_concepts개 = Mean scores
        Returns:
            logits: [B, num_classes]
        """
        # 1. Feature Interaction
        max_s = concept_scores[:, :self.num_concepts]    # [B, 4]
        mean_s = concept_scores[:, self.num_concepts:]   # [B, 4]
        
        # Pairwise interaction: max_i * mean_j for all (i,j)
        # [B, 4, 1] × [B, 1, 4] → [B, 4, 4] → [B, 16]
        interaction = torch.bmm(
            max_s.unsqueeze(2),    # [B, 4, 1]
            mean_s.unsqueeze(1)    # [B, 1, 4]
        ).view(concept_scores.size(0), -1)  # [B, 16]
        
        # Concat: [B, 8] + [B, 16] = [B, 24]
        x = torch.cat([concept_scores, interaction], dim=1)
        
        # 2. Non-linear processing
        x = self.input_proj(x)     # [B, 128]
        x = self.res_block(x)      # [B, 128] (residual)
        logits = self.classifier(x) # [B, 5]
        
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_ordinal_bias(self, num_classes):
        """
        최종 분류 layer의 bias를 ordinal 순서로 초기화.
        Grade 0이 가장 활성화되기 쉽게 (DR 데이터는 정상이 많으므로),
        Grade 4로 갈수록 초기 bias가 낮아지도록 설정.
        """
        final_layer = self.classifier[-1]
        if final_layer.bias is not None:
            with torch.no_grad():
                # 등차적으로 감소하는 bias
                bias_vals = torch.linspace(0.5, -0.5, num_classes)
                final_layer.bias.copy_(bias_vals)


class ResidualBlock(nn.Module):
    """
    Pre-LayerNorm Residual Block
    
    x → LayerNorm → Linear → GELU → Dropout → Linear → + x
    """
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self._init_weights()
    
    def forward(self, x):
        return x + self.block(x)
    
    def _init_weights(self):
        # 마지막 Linear를 near-zero로 초기화 → 초기에 identity mapping
        layers = [m for m in self.block if isinstance(m, nn.Linear)]
        if layers:
            nn.init.zeros_(layers[-1].weight)
            if layers[-1].bias is not None:
                nn.init.zeros_(layers[-1].bias)


# --- 테스트 ---
if __name__ == "__main__":
    head = OrdinalRegressionHead(input_dim=8, num_classes=5)
    
    # 파라미터 수 비교
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Parameters: {n_params:,}")  # ~25K (기존 ~1.4K 대비 18x)
    
    # Hybrid Score 시뮬레이션
    # [max_EX, max_HE, max_MA, max_SE, mean_EX, mean_HE, mean_MA, mean_SE]
    severe = torch.tensor([[15.0, 12.0, 8.0, 10.0,  5.0, 4.0, 3.0, 4.0]])
    normal = torch.tensor([[-2.0, -3.0, -1.0, -2.0, -1.0, -1.5, -0.5, -1.0]])
    
    batch = torch.cat([severe, normal], dim=0)
    
    head.set_session_mode('base')
    logits = head(batch)
    print(f"Logits shape: {logits.shape}")  # [2, 5]
    print(f"Severe prediction: Grade {logits[0].argmax().item()}")
    print(f"Normal prediction: Grade {logits[1].argmax().item()}")
    
    # Interaction features 확인
    print(f"\nInput dim: {head.input_dim}")
    print(f"Interaction dim: {head.interaction_dim}")  # 8 + 16 = 24