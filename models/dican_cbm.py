import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .projector import ConceptProjector
from .prototypes import PrototypeBank
from .head import OrdinalRegressionHead
from .seg_decoder import SegDecoder


class DICAN_CBM(nn.Module):
    """
    DICAN - Pixel-Level Segmentation Supervision 버전
    
    [핵심 변경: SegHead → SegDecoder]
    
    Before: Backbone [2048,7,7] → Conv1x1 → [4,7,7] (32×32 영역 단위)
            → 마스크를 7×7로 줄여서 비교 → 디테일 소실
    
    After:  Backbone multi-scale features
              layer1 [256, 56,56] ─┐
              layer2 [512, 28,28] ─┤  skip connections
              layer3 [1024,14,14] ─┤
              layer4 [2048, 7, 7] ─┘
            → SegDecoder (U-Net 스타일) → [4, 224, 224]
            → 원본 해상도 마스크와 직접 비교 → pixel-level 학습
    
    MA(미세동맥류)처럼 수 픽셀짜리 병변도
    224×224에서는 존재가 명확하게 드러남.
    
    [Phase 구조 - 동일]
    1-A: Backbone + TempHead + SegDecoder 학습 (★ pixel-level seg)
    1-B: Backbone Freeze → Masked GAP → Prototype
    1-C: Head 학습
    2:   Projector 학습
    """
    
    def __init__(self, num_concepts=4, num_classes=5, feature_dim=2048):
        super(DICAN_CBM, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = 'pretrain'

        # 1. Backbone (Multi-Scale 지원)
        self.backbone = ResNetBackbone(pretrained=True)
        
        # 2. Projector (Phase 2)
        self.projector = ConceptProjector(input_dim=feature_dim)
        
        # 3. Prototype Bank (Hybrid Pooling)
        self.prototypes = PrototypeBank(
            num_concepts=num_concepts, 
            feature_dim=feature_dim
        )
        
        # 4. CBM Head (Phase 1-C, Phase 2)
        score_dim = self.prototypes.get_score_dim()
        self.head = OrdinalRegressionHead(
            input_dim=score_dim, 
            num_classes=num_classes
        )
        
        # 5. Temp Classification Head (Phase 1-A, 이후 폐기)
        self.temp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # =============================================
        # 6. Pixel-Level Segmentation Decoder [★ 핵심]
        #
        #    7×7 SegHead 대신 U-Net 스타일 Decoder 사용
        #    Backbone의 layer1~4 skip connection으로
        #    224×224까지 복원하여 pixel 단위로 병변 학습
        #
        #    이 Decoder의 gradient가 Backbone 전체를 관통하여
        #    "layer1의 56×56에서도 병변 경계를 인식"하게 만듦
        # =============================================
        self.seg_decoder = SegDecoder(num_concepts=num_concepts)
        
        self._init_temp_head()

    def _init_temp_head(self):
        for m in self.temp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        
        # ===========================================================
        # Phase 1-A: Pretrain (Classification + Pixel-Level Segmentation)
        #
        # Backbone에서 multi-scale feature를 뽑고:
        #   (a) layer4 → GAP → TempHead → 등급 분류
        #   (b) layer1~4 → SegDecoder → [B,4,224,224] pixel 예측
        #
        # Decoder의 gradient가 layer1~4 모두를 통과하므로
        # Backbone의 모든 레벨에서 병변 인식 능력이 형성됨
        # ===========================================================
        if self.mode == 'pretrain':
            # Multi-scale feature 추출
            feat4, multi_scale = self.backbone(x, return_multi_scale=True)
            
            # (a) Classification
            cls_logits = self.temp_head(feat4)  # [B, 5]
            
            # (b) Pixel-level Segmentation
            seg_pred = self.seg_decoder(multi_scale)  # [B, 4, 224, 224] ★
            
            return {
                "logits": cls_logits,
                "seg_pred": seg_pred,  # 224×224 pixel-level
                "concept_scores": None,
                "features": feat4,
                "spatial_sim_map": None
            }
        
        # ===========================================================
        # Phase 1-C: Head Training
        # ===========================================================
        elif self.mode == 'head_train':
            with torch.no_grad():
                raw_features = self.backbone(x)
            
            concept_scores, spatial_sim_map = self.prototypes(raw_features)
            logits = self.head(concept_scores)
            
            return {
                "logits": logits,
                "seg_pred": None,
                "concept_scores": concept_scores,
                "features": raw_features.detach(),
                "spatial_sim_map": spatial_sim_map.detach()
            }
        
        # ===========================================================
        # Phase 2: Incremental
        # ===========================================================
        elif self.mode == 'incremental':
            with torch.no_grad():
                raw_features = self.backbone(x)
            
            aligned_features = self.projector(raw_features)
            concept_scores, spatial_sim_map = self.prototypes(aligned_features)
            logits = self.head(concept_scores)
            
            return {
                "logits": logits,
                "seg_pred": None,
                "concept_scores": concept_scores,
                "features": aligned_features,
                "spatial_sim_map": spatial_sim_map
            }
        
        # ===========================================================
        # Evaluation
        # ===========================================================
        elif self.mode == 'eval':
            with torch.no_grad():
                raw_features = self.backbone(x)
                aligned_features = self.projector(raw_features)
                concept_scores, spatial_sim_map = self.prototypes(aligned_features)
                logits = self.head(concept_scores)
            
            return {
                "logits": logits,
                "seg_pred": None,
                "concept_scores": concept_scores,
                "features": aligned_features,
                "spatial_sim_map": spatial_sim_map
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_session_mode(self, mode):
        valid_modes = ['pretrain', 'extract', 'head_train', 'incremental', 'eval']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid: {valid_modes}")
        
        print(f"\n[*] DICAN mode → {mode.upper()}")
        self.mode = mode
        
        if mode == 'pretrain':
            # Backbone + TempHead + SegDecoder 학습
            self.backbone.set_session_mode('base')
            for p in self.temp_head.parameters():
                p.requires_grad = True
            self.temp_head.train()
            for p in self.seg_decoder.parameters():
                p.requires_grad = True
            self.seg_decoder.train()
            # 나머지 Freeze
            for p in self.projector.parameters():
                p.requires_grad = False
            self.projector.eval()
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()
            
            print("   → Backbone:    Trainable")
            print("   → TempHead:    Trainable (Classification)")
            print("   → SegDecoder:  Trainable (224×224 Pixel-Level Seg)") 
            print("   → Projector/CBM Head: Frozen")
            
        elif mode == 'extract':
            self.backbone.set_session_mode('incremental')
            self.eval()
            print("   → All Frozen (Prototype extraction)")
            
        elif mode == 'head_train':
            self.backbone.set_session_mode('incremental')
            for p in self.projector.parameters():
                p.requires_grad = False
            self.projector.eval()
            for p in self.head.parameters():
                p.requires_grad = True
            self.head.train()
            self.prototypes.logit_scale.requires_grad = True
            for p in self.seg_decoder.parameters():
                p.requires_grad = False
            self.seg_decoder.eval()
            for p in self.temp_head.parameters():
                p.requires_grad = False
            self.temp_head.eval()
            
            print("   → Backbone:  Frozen (lesion-aware)")
            print("   → CBM Head:  Trainable")
            print("   → logit_scale: Trainable")
            
        elif mode == 'incremental':
            self.backbone.set_session_mode('incremental')
            self.projector.set_session_mode('incremental')
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()
            self.prototypes.logit_scale.requires_grad = False
            print("   → Projector: Trainable")
            print("   → All else:  Frozen")
            
        elif mode == 'eval':
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def get_trainable_params(self):
        if self.mode == 'pretrain':
            params = (list(self.backbone.parameters()) + 
                     list(self.temp_head.parameters()) + 
                     list(self.seg_decoder.parameters()))
        elif self.mode == 'head_train':
            params = list(self.head.parameters()) + [self.prototypes.logit_scale]
        elif self.mode == 'incremental':
            params = list(self.projector.parameters())
        else:
            params = []
        return [p for p in params if p.requires_grad]