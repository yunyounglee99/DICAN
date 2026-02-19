"""
DICAN Concept Projector - Multi-Variant Support
=================================================
3가지 Projector 아키텍처를 --projector_type 인자로 선택 가능:

1. 'lora' (기존, default):
   Low-Rank Residual Projector (2048→rank→2048)
   - 파라미터: ~262K (rank=64)
   - Zero-init residual로 초기 identity 보장
   - Few-shot에서 안정적 (BN 없음)

2. 'linear_1layer':
   1-Layer Linear CBM Projector (2048→2048)
   - 파라미터: ~4.2M
   - 가장 단순한 CBM 스타일 변환
   - Residual connection으로 identity 초기화

3. 'linear_2layer':
   2-Layer Linear CBM Projector (2048→hidden→2048)
   - 파라미터: ~2.1M (hidden=512)
   - BN 포함 (CBM 표준), Few-shot에서 불안정할 수 있음
   - Residual connection + Zero-init

[Ablation 실험 사용법]
  python train/train.py --projector_type lora       # LoRA (Ours)
  python train/train.py --projector_type linear_1layer  # 1-Layer CBM
  python train/train.py --projector_type linear_2layer  # 2-Layer CBM
"""

import torch
import torch.nn as nn
import math


# =================================================================
# Factory Function
# =================================================================
def build_projector(projector_type='lora', input_dim=2048, **kwargs):
    """
    Projector 아키텍처 팩토리.
    
    Args:
        projector_type: 'lora' | 'linear_1layer' | 'linear_2layer'
        input_dim: Backbone 출력 채널 수
        **kwargs: 아키텍처별 추가 인자
            - lora:          rank (default 64)
            - linear_1layer: (추가 인자 없음)
            - linear_2layer: hidden_dim (default 512)
    
    Returns:
        ConceptProjector 인스턴스 (공통 인터페이스)
    """
    if projector_type == 'lora':
        rank = kwargs.get('rank', 64)
        return LoRAProjector(input_dim=input_dim, rank=rank)
    elif projector_type == 'linear_1layer':
        return Linear1LayerProjector(input_dim=input_dim)
    elif projector_type == 'linear_2layer':
        hidden_dim = kwargs.get('hidden_dim', 512)
        return Linear2LayerProjector(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(
            f"Unknown projector_type: '{projector_type}'. "
            f"Choose from: 'lora', 'linear_1layer', 'linear_2layer'"
        )


# =================================================================
# Base Class (공통 인터페이스)
# =================================================================
class ConceptProjector(nn.Module):
    """
    모든 Projector가 공유하는 인터페이스.
    DICAN_CBM과의 호환성을 위해 아래 메서드를 반드시 구현:
      - forward(x) → [B, C, H, W]
      - set_session_mode(mode)
      - mode 속성 ('base' / 'incremental')
    """
    
    def __init__(self, input_dim=2048):
        super(ConceptProjector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.mode = 'base'
        self._projector_type = 'base'  # 서브클래스에서 오버라이드

    @property
    def projector_type(self):
        return self._projector_type

    def forward(self, x):
        raise NotImplementedError

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

    def get_param_count(self):
        """학습 가능한 파라미터 수 (layer 부분만)"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (f"type={self._projector_type}, "
                f"dim={self.input_dim}, "
                f"params={self.get_param_count():,}")


# =================================================================
# Variant 1: LoRA Projector (기존, Our Method)
# =================================================================
class LoRAProjector(ConceptProjector):
    """
    Low-Rank Residual Projector (기존 코드와 동일).
    
    구조: x + Conv(rank→C, GELU(Conv(C→rank, x)))
    
    [장점]
    - 파라미터 32배 축소 (rank=64 기준)
    - BN 없음 → Few-shot 안정
    - Zero-init → 초기 identity 보장
    
    [파라미터]
    rank=32:  131K (64배 축소)
    rank=64:  262K (32배 축소) ← 권장
    rank=128: 524K (16배 축소)
    """

    def __init__(self, input_dim=2048, rank=64):
        super(LoRAProjector, self).__init__(input_dim)
        self._projector_type = 'lora'
        self.rank = rank

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, rank, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(rank, input_dim, kernel_size=1, bias=False),
        )
        
        self._init_weights()

    def forward(self, x):
        if self.mode == 'base':
            return x
        return x + self.layer(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 마지막 Conv를 0으로 → 초기 output = identity
        last_conv = self.layer[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)

    def extra_repr(self):
        return (f"type=lora, rank={self.rank}, "
                f"dim={self.input_dim}, "
                f"params={self.get_param_count():,}")


# =================================================================
# Variant 2: 1-Layer Linear CBM Projector
# =================================================================
class Linear1LayerProjector(ConceptProjector):
    """
    1-Layer Linear CBM Projector.
    
    구조: x + Conv1x1(C→C, x)  (Residual)
    
    [특징]
    - CBM 논문에서 가장 일반적인 Linear Projection
    - Full-rank 변환이므로 표현력은 높지만 파라미터가 많음
    - BN 없음 (Few-shot 호환)
    - Zero-init으로 초기 identity 보장
    
    [파라미터]
    2048 × 2048 = ~4.2M
    """

    def __init__(self, input_dim=2048):
        super(Linear1LayerProjector, self).__init__(input_dim)
        self._projector_type = 'linear_1layer'

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
        )
        
        self._init_weights()

    def forward(self, x):
        if self.mode == 'base':
            return x
        return x + self.layer(x)

    def _init_weights(self):
        # Zero-init → residual이 identity
        for m in self.layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)

    def extra_repr(self):
        return (f"type=linear_1layer, "
                f"dim={self.input_dim}, "
                f"params={self.get_param_count():,}")


# =================================================================
# Variant 3: 2-Layer Linear CBM Projector
# =================================================================
class Linear2LayerProjector(ConceptProjector):
    """
    2-Layer Linear CBM Projector with Bottleneck.
    
    구조: x + BN(Conv1x1(H→C, GELU(BN(Conv1x1(C→H, x)))))
    
    [특징]
    - CBM 논문의 표준 Multi-Layer Projection
    - Bottleneck으로 파라미터 절감
    - BatchNorm 포함 → 정규화 효과 있지만 Few-shot에서 불안정 가능
    - Zero-init으로 초기 identity 보장
    
    [파라미터]
    hidden=512: 2048*512 + 512*2048 + BN ≈ 2.1M
    hidden=256: 2048*256 + 256*2048 + BN ≈ 1.05M
    
    [vs LoRA 차이점]
    - LoRA:    BN 없음,  rank=64  → 262K
    - 2Layer:  BN 있음,  hidden=512 → 2.1M  (8배 더 큼)
    → Ablation에서 BN 유무 + 파라미터 효율성 비교 가능
    """

    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Linear2LayerProjector, self).__init__(input_dim)
        self._projector_type = 'linear_2layer'
        self.hidden_dim = hidden_dim

        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_dim),
        )
        
        self._init_weights()

    def forward(self, x):
        if self.mode == 'base':
            return x
        return x + self.layer(x)

    def _init_weights(self):
        for m in self.layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 마지막 Conv를 0으로 → 초기 output = identity
        # (BN은 weight=1, bias=0이므로 0*1+0=0 → identity 유지)
        last_conv = None
        for m in self.layer.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            nn.init.zeros_(last_conv.weight)

    def extra_repr(self):
        return (f"type=linear_2layer, hidden={self.hidden_dim}, "
                f"dim={self.input_dim}, "
                f"params={self.get_param_count():,}")


# =================================================================
# 테스트
# =================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Projector Variant Comparison")
    print("=" * 60)
    
    dummy = torch.randn(2, 2048, 7, 7)
    
    variants = [
        ('lora',          {'rank': 64}),
        ('linear_1layer', {}),
        ('linear_2layer', {'hidden_dim': 512}),
    ]
    
    for ptype, kwargs in variants:
        proj = build_projector(ptype, input_dim=2048, **kwargs)
        
        # Base mode (identity)
        proj.set_session_mode('base')
        out_base = proj(dummy)
        assert torch.allclose(out_base, dummy), f"{ptype}: Base mode should be identity!"
        
        # Incremental mode (residual, initially near-identity)
        proj.set_session_mode('incremental')
        out_inc = proj(dummy)
        diff = (out_inc - dummy).abs().mean().item()
        
        n_params = proj.get_param_count()
        trainable = sum(p.numel() for p in proj.parameters() if p.requires_grad)
        
        print(f"\n[{ptype}]")
        print(f"  Parameters:      {n_params:>10,}")
        print(f"  Trainable (inc): {trainable:>10,}")
        print(f"  Output shape:    {out_inc.shape}")
        print(f"  Init diff (should be ~0): {diff:.6f}")
        print(f"  repr: {proj.extra_repr()}")
    
    print("\n" + "=" * 60)
    print("All variants passed identity check!")
    print("=" * 60)