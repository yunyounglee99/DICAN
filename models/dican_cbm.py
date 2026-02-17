import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .projector import ConceptProjector
from .prototypes import PrototypeBank
from .head import OrdinalRegressionHead


class DICAN_CBM(nn.Module):
    """
    DICAN - Phase 분리 + Segmentation Supervision 버전
    
    [핵심 설계]
    Phase 1-A에서 Backbone이 두 가지를 동시에 학습:
      (a) Classification: "이 이미지의 DR 등급은 무엇인가?" (GAP → TempHead)
      (b) Segmentation:   "병변이 7×7 어디에 있는가?"      (SegHead → BCE with mask)
    
    → Backbone이 "병변 위치 인식(Spatial Awareness)"을 학습한 상태에서 고정됨
    → Phase 1-B에서 Masked GAP로 추출하는 prototype의 품질이 극적으로 향상됨
    
    [Phase 구조]
    Phase 1-A (pretrain):  Backbone + TempHead + SegHead 학습
    Phase 1-B (extract):   Backbone Freeze → Masked Pooling → Prototype 구축
    Phase 1-C (head_train): Backbone + Prototype Freeze → CBM Head 학습
    Phase 2   (incremental): Projector만 학습
    """
    
    def __init__(self, num_concepts=4, num_classes=5, feature_dim=2048):
        super(DICAN_CBM, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = 'pretrain'

        # 1. Backbone (ResNet-50)
        self.backbone = ResNetBackbone(pretrained=True)
        
        # 2. Concept Projector (Phase 2용)
        self.projector = ConceptProjector(input_dim=feature_dim)
        
        # 3. Prototype Bank (Hybrid Pooling)
        self.prototypes = PrototypeBank(
            num_concepts=num_concepts, 
            feature_dim=feature_dim
        )
        
        # 4. CBM Reasoning Head (Phase 1-C, Phase 2)
        score_dim = self.prototypes.get_score_dim()
        self.head = OrdinalRegressionHead(
            input_dim=score_dim, 
            num_classes=num_classes
        )
        
        # 5. Temporary Classification Head (Phase 1-A, 이후 폐기)
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
        # 6. Segmentation Head (Phase 1-A) [★ 핵심]
        #
        #    Backbone [B, 2048, 7, 7] → SegHead → [B, 4, 7, 7]
        #    각 Concept(EX, HE, MA, SE)별 공간 활성화 맵 예측
        #
        #    이것이 Backbone에게 가르치는 것:
        #    "이 7×7 좌표에 출혈이 있다/없다"
        #    "이 7×7 좌표에 삼출물이 있다/없다"
        #
        #    → feature map의 각 픽셀이 병변 유무를 인코딩하게 됨
        #    → Phase 1-B의 Masked GAP에서 추출되는 벡터가
        #      "진짜 병변 특징"을 담게 됨
        # =============================================
        self.seg_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_concepts, kernel_size=1, bias=True)
        )
        
        self._init_auxiliary_heads()

    def _init_auxiliary_heads(self):
        for module in [self.temp_head, self.seg_head]:
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        
        # ===========================================================
        # Phase 1-A: Pretrain
        #   Backbone → feature [B,2048,7,7]
        #              ├→ GAP → TempHead → cls_logits    (등급 분류)
        #              └→ SegHead → seg_pred [B,4,7,7]   (병변 위치)
        # ===========================================================
        if self.mode == 'pretrain':
            raw_features = self.backbone(x)
            cls_logits = self.temp_head(raw_features)
            seg_pred = self.seg_head(raw_features)
            
            return {
                "logits": cls_logits,
                "seg_pred": seg_pred,
                "concept_scores": None,
                "features": raw_features,
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
            self.backbone.set_session_mode('base')
            for p in self.temp_head.parameters():
                p.requires_grad = True
            self.temp_head.train()
            for p in self.seg_head.parameters():
                p.requires_grad = True
            self.seg_head.train()
            for p in self.projector.parameters():
                p.requires_grad = False
            self.projector.eval()
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()
            
            print("   → Backbone: Trainable")
            print("   → TempHead: Trainable (Classification)")
            print("   → SegHead:  Trainable (Lesion Localization)")
            print("   → Projector/CBM Head/Prototypes: Frozen")
            
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
            for p in self.seg_head.parameters():
                p.requires_grad = False
            self.seg_head.eval()
            for p in self.temp_head.parameters():
                p.requires_grad = False
            self.temp_head.eval()
            
            print("   → Backbone: Frozen (lesion-aware features locked)")
            print("   → Prototypes: Fixed (logit_scale trainable)")
            print("   → CBM Head: Trainable")
            
        elif mode == 'incremental':
            self.backbone.set_session_mode('incremental')
            self.projector.set_session_mode('incremental')
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()
            self.prototypes.logit_scale.requires_grad = False
            print("   → Projector: Trainable")
            print("   → Backbone + Head + Prototypes: Frozen")
            
        elif mode == 'eval':
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def get_trainable_params(self):
        if self.mode == 'pretrain':
            params = (list(self.backbone.parameters()) + 
                     list(self.temp_head.parameters()) + 
                     list(self.seg_head.parameters()))
        elif self.mode == 'head_train':
            params = list(self.head.parameters()) + [self.prototypes.logit_scale]
        elif self.mode == 'incremental':
            params = list(self.projector.parameters())
        else:
            params = []
        return [p for p in params if p.requires_grad]