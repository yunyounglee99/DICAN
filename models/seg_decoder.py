import torch
import torch.nn as nn
import torch.nn.functional as F


class SegDecoder(nn.Module):
    """
    Pixel-Level Segmentation Decoder (Phase 1-A 전용)
    
    [목적]
    Backbone의 7×7 feature를 224×224 pixel level까지 복원하여
    "각 픽셀에 EX/HE/MA/SE가 있는가?"를 예측.
    
    이 Decoder가 Backbone에게 가르치는 것:
    - 7×7의 각 위치가 "정확히 어떤 병변의 특징"을 담아야 하는지
    - Skip Connection을 통해 고해상도의 경계선 정보까지 학습
    
    [구조: U-Net 스타일 Decoder]
    
    Encoder (ResNet-50, 기존 Backbone)     Decoder (이 모듈)
    ─────────────────────────             ──────────────────
    layer4: [2048,  7,  7] ─────────────→ Up → [256, 14, 14]
                                                    ↑ concat
    layer3: [1024, 14, 14] ─────────────→        [256, 14, 14]
                                          Conv → [128, 14, 14]
                                          Up   → [128, 28, 28]
                                                    ↑ concat
    layer2: [ 512, 28, 28] ─────────────→        [128, 28, 28]
                                          Conv → [ 64, 28, 28]
                                          Up   → [ 64, 56, 56]
                                                    ↑ concat
    layer1: [ 256, 56, 56] ─────────────→        [ 64, 56, 56]
                                          Conv → [ 32, 56, 56]
                                          Up   → [ 32,224,224]
                                          Final→ [  4,224,224]  ← Pixel-level 출력
    
    [Phase 1-A 이후 폐기]
    이 Decoder는 Backbone에 병변 인식 능력을 주입하기 위한 보조 모듈이며,
    Phase 1-B 이후에는 사용하지 않음.
    """
    
    def __init__(self, num_concepts=4):
        super(SegDecoder, self).__init__()
        
        # =============================================
        # Decoder Block 1: 7×7 → 14×14
        # Input: layer4(2048) upsampled + layer3(1024) = 3072ch
        # =============================================
        self.up4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),  # 채널 압축
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = self._make_decoder_block(256 + 1024, 256)
        
        # =============================================
        # Decoder Block 2: 14×14 → 28×28
        # Input: dec3(256) upsampled + layer2(512) = 768ch
        # =============================================
        self.dec2 = self._make_decoder_block(256 + 512, 128)
        
        # =============================================
        # Decoder Block 3: 28×28 → 56×56
        # Input: dec2(128) upsampled + layer1(256) = 384ch
        # =============================================
        self.dec1 = self._make_decoder_block(128 + 256, 64)
        
        # =============================================
        # Final: 56×56 → 224×224
        # 4× Upsample + 1×1 Conv
        # =============================================
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_concepts, kernel_size=1)
            # 출력: [B, 4, 56, 56] → bilinear upsample to 224×224 in forward
        )
        
        self._init_weights()
    
    def _make_decoder_block(self, in_channels, out_channels):
        """
        Skip Connection 합류 후 처리하는 Conv Block
        [핵심] 3×3 Conv를 2개 쌓아서 skip feature와 upsampled feature를
        효과적으로 융합 (U-Net 표준 구조)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, multi_scale):
        """
        Args:
            multi_scale (dict): Backbone에서 반환된 multi-scale features
                'layer1': [B,  256, 56, 56]
                'layer2': [B,  512, 28, 28]
                'layer3': [B, 1024, 14, 14]
                'layer4': [B, 2048,  7,  7]
        
        Returns:
            seg_pred: [B, 4, 224, 224] (pixel-level logits, Sigmoid 전)
        """
        f1 = multi_scale['layer1']  # [B, 256, 56, 56]
        f2 = multi_scale['layer2']  # [B, 512, 28, 28]
        f3 = multi_scale['layer3']  # [B, 1024, 14, 14]
        f4 = multi_scale['layer4']  # [B, 2048, 7, 7]
        
        # Stage 1: 7→14
        x = self.up4(f4)                                        # [B, 256, 7, 7]
        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)  # [B, 256, 14, 14]
        x = torch.cat([x, f3], dim=1)                           # [B, 1280, 14, 14]
        x = self.dec3(x)                                         # [B, 256, 14, 14]
        
        # Stage 2: 14→28
        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)  # [B, 256, 28, 28]
        x = torch.cat([x, f2], dim=1)                           # [B, 768, 28, 28]
        x = self.dec2(x)                                         # [B, 128, 28, 28]
        
        # Stage 3: 28→56
        x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)  # [B, 128, 56, 56]
        x = torch.cat([x, f1], dim=1)                           # [B, 384, 56, 56]
        x = self.dec1(x)                                         # [B, 64, 56, 56]
        
        # Stage 4: 56→224
        x = self.final(x)                                        # [B, 4, 56, 56]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # [B, 4, 224, 224]
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# --- 테스트 ---
if __name__ == "__main__":
    decoder = SegDecoder(num_concepts=4)
    
    # 더미 multi-scale features (ResNet-50 출력 시뮬레이션)
    ms = {
        'layer1': torch.randn(2, 256, 56, 56),
        'layer2': torch.randn(2, 512, 28, 28),
        'layer3': torch.randn(2, 1024, 14, 14),
        'layer4': torch.randn(2, 2048, 7, 7)
    }
    
    seg_pred = decoder(ms)
    print(f"Seg Prediction: {seg_pred.shape}")  # [2, 4, 224, 224]
    
    # 파라미터 수 확인
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"SegDecoder Parameters: {n_params:,}")  # ~3-5M 정도
    
    # Pixel-level Loss 시뮬레이션
    gt_mask = torch.randint(0, 2, (2, 4, 224, 224)).float()
    loss = F.binary_cross_entropy_with_logits(seg_pred, gt_mask)
    print(f"BCE Loss: {loss.item():.4f}")