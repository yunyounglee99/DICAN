import torch
import torch.nn as nn
import math

class ConceptProjector(nn.Module):
    """
    DICAN Low-Rank Concept Projector
    
    [기존 대비 변경사항]
    1. Bottleneck 도입: 2048→rank→2048 (rank=64 기준 파라미터 32배 축소)
    2. BatchNorm 제거: Few-shot 불안정 해소
    3. Residual + Zero Init 유지: 기존의 성능 장점 보존
    
    [파라미터 비교]
    기존:  Conv(2048→2048) + BN + Conv(2048→2048) = ~8.4M
    변경:  Conv(2048→64)   +      Conv(64→2048)   = ~262K  (32배 축소)
    
    [왜 이것이 작동하는가]
    도메인 간 차이(DDR→APTOS)는 2048차원 전체가 아니라,
    저차원 부분공간(subspace)에 집중되어 있음.
    rank=64는 전체의 3%만 사용하지만, 도메인 shift 보정에는 충분.
    """

    def __init__(self, input_dim=2048, rank=64):
        """
        Args:
            input_dim (int): Backbone 출력 채널 수 (ResNet50: 2048)
            rank (int): Bottleneck 차원. 
                        rank=32:  131K params (64배 축소)
                        rank=64:  262K params (32배 축소) ← 권장
                        rank=128: 524K params (16배 축소)
        """
        super(ConceptProjector, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.rank = rank
        self.mode = 'base'

        # ─────────────────────────────────────────────
        # Low-Rank Residual Block
        # 
        # 기존:  Conv(2048→2048) + BN + GELU + Conv(2048→2048)
        # 변경:  Conv(2048→rank) + GELU + Conv(rank→2048)
        #
        # BN 제거 이유:
        #   - Few-shot (25샘플, batch 4~8)에서 running stats 불안정
        #   - Residual 구조이므로 입력 분포가 이미 보존됨
        #   - One-Shot DIL 논문: "BN statistics are the primary bottleneck"
        # ─────────────────────────────────────────────
        self.layer = nn.Sequential(
            # Down: 2048 → rank (정보 압축)
            nn.Conv2d(input_dim, rank, kernel_size=1, bias=False),
            # 비선형성: 중간에서만 적용 (출력에는 적용 안 함)
            nn.GELU(),
            # Up: rank → 2048 (원래 차원 복원)
            nn.Conv2d(rank, input_dim, kernel_size=1, bias=False),
        )
        
        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: [B, 2048, 7, 7] from Frozen Backbone
        Returns:
            [B, 2048, 7, 7] aligned features
        
        Base mode:        return x (identity, projector 비활성)
        Incremental mode: return x + self.layer(x) (residual 보정)
        
        [초기 상태]
        Up conv가 0으로 초기화되어 있으므로:
        self.layer(x) = Up(GELU(Down(x))) = 0
        → output = x + 0 = x (완벽한 identity)
        """
        if self.mode == 'base':
            return x
        return x + self.layer(x)

    def set_session_mode(self, mode):
        self.mode = mode
        if mode == 'base':
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        elif mode == 'incremental':
            for param in self.parameters():
                param.requires_grad = True
            self.train()

    def _init_weights(self):
        """
        [초기화 전략 — 기존과 동일한 원리]
        
        Down conv: Kaiming init → 입력 feature를 rank 차원으로 잘 압축
        Up conv:   Zero init    → 초기 출력 = 0 → residual이 identity가 됨
        
        이것이 기존 2-layer의 성능 핵심이었고, 그대로 유지함.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # ★ 마지막 Conv(Up)를 0으로 초기화 → 초기 output = identity
        last_conv = self.layer[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)