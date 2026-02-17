import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    DICAN Backbone - Multi-Scale Feature Extraction 지원
    
    [변경사항]
    기존: nn.Sequential로 묶어서 최종 7×7만 출력
    변경: ResNet의 4개 Stage를 분리하여 중간 feature도 반환 가능
    
    [ResNet-50 Feature Map 해상도]
    stem (conv1+bn+pool): 224 → 56×56
    layer1:               56  → 56×56,  256 channels
    layer2:               56  → 28×28,  512 channels
    layer3:               28  → 14×14, 1024 channels
    layer4:               14  →  7×7,  2048 channels
    
    Pixel-level Segmentation을 위해 layer1~4의 feature를 모두 반환하여
    Decoder가 skip connection으로 224×224까지 복원할 수 있게 함.
    """

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Stem: conv1 → bn1 → relu → maxpool (224 → 56)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # 4개 Residual Stage 분리
        self.layer1 = resnet.layer1   # 56→56,   256ch
        self.layer2 = resnet.layer2   # 56→28,   512ch
        self.layer3 = resnet.layer3   # 28→14,  1024ch
        self.layer4 = resnet.layer4   # 14→7,   2048ch

        self.output_dim = 2048

    def forward(self, x, return_multi_scale=False):
        """
        Args:
            x: [B, 3, 224, 224]
            return_multi_scale: True면 중간 feature도 함께 반환 (SegDecoder용)
        
        Returns:
            return_multi_scale=False:
                features: [B, 2048, 7, 7] (기존과 동일)
            
            return_multi_scale=True:
                features: [B, 2048, 7, 7]
                multi_scale: dict {
                    'layer1': [B,  256, 56, 56],
                    'layer2': [B,  512, 28, 28],
                    'layer3': [B, 1024, 14, 14],
                    'layer4': [B, 2048,  7,  7]
                }
        """
        x0 = self.stem(x)      # [B, 64, 56, 56]
        x1 = self.layer1(x0)   # [B, 256, 56, 56]
        x2 = self.layer2(x1)   # [B, 512, 28, 28]
        x3 = self.layer3(x2)   # [B, 1024, 14, 14]
        x4 = self.layer4(x3)   # [B, 2048, 7, 7]

        if return_multi_scale:
            return x4, {
                'layer1': x1,
                'layer2': x2,
                'layer3': x3,
                'layer4': x4
            }
        
        return x4

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

    def get_feature_dim(self):
        return self.output_dim


if __name__ == "__main__":
    backbone = ResNetBackbone(pretrained=True)
    dummy = torch.randn(2, 3, 224, 224)
    
    # 기존 호환 모드
    feat = backbone(dummy)
    print(f"[Standard] Output: {feat.shape}")  # [2, 2048, 7, 7]
    
    # Multi-scale 모드
    feat, ms = backbone(dummy, return_multi_scale=True)
    for name, tensor in ms.items():
        print(f"[MultiScale] {name}: {tensor.shape}")
    # layer1: [2, 256, 56, 56]
    # layer2: [2, 512, 28, 28]
    # layer3: [2, 1024, 14, 14]
    # layer4: [2, 2048, 7, 7]