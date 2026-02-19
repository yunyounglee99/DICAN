import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetBackbone
from .projector import build_projector, ConceptProjector
from .prototypes import PrototypeBank
from .head import OrdinalRegressionHead
from .seg_decoder import SegDecoder


class DICAN_CBM(nn.Module):
    """
    DICAN - Multi-Cluster Prototype + Enhanced Head + Multi-Variant Projector
    
    [핵심 변경사항]
    1. PrototypeBank: Multi-Cluster (k=3) + Orthogonality Loss
    2. OrdinalRegressionHead: 12-dim 입력 + Residual MLP
    3. Phase 1-A: Orthogonality Loss 추가 (seg_loss와 함께)
    4. ★ Projector: --projector_type으로 아키텍처 선택 가능
       - 'lora' (default, Our Method)
       - 'linear_1layer' (1-Layer CBM baseline)
       - 'linear_2layer' (2-Layer CBM baseline)
    """
    
    def __init__(self, num_concepts=4, num_classes=5, feature_dim=2048, 
                 num_clusters=3, projector_type='lora', projector_kwargs=None):
        """
        Args:
            num_concepts: 병변 Concept 수 (EX, HE, MA, SE)
            num_classes: DR Grade 수 (0~4)
            feature_dim: Backbone 출력 차원
            num_clusters: Prototype 클러스터 수
            projector_type: 'lora' | 'linear_1layer' | 'linear_2layer'
            projector_kwargs: Projector 아키텍처별 추가 인자 (dict)
                - lora: {'rank': 64}
                - linear_2layer: {'hidden_dim': 512}
        """
        super(DICAN_CBM, self).__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = 'pretrain'
        self.projector_type = projector_type

        # 1. Backbone
        self.backbone = ResNetBackbone(pretrained=True)
        
        # 2. ★ Projector (팩토리로 생성)
        if projector_kwargs is None:
            projector_kwargs = {}
        self.projector = build_projector(
            projector_type=projector_type,
            input_dim=feature_dim,
            **projector_kwargs
        )
        print(f"[DICAN] Projector: {self.projector.extra_repr()}")
        
        # 3. Multi-Cluster Prototype Bank
        self.prototypes = PrototypeBank(
            num_concepts=num_concepts, 
            feature_dim=feature_dim,
            num_clusters=num_clusters
        )
        
        # 4. Enhanced CBM Head (12-dim input)
        score_dim = self.prototypes.get_score_dim()
        self.head = OrdinalRegressionHead(
            input_dim=score_dim, 
            num_classes=num_classes
        )
        
        # 5. Temp Head (Phase 1-A)
        self.temp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 6. Seg Decoder
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
        
        if self.mode == 'pretrain':
            feat4, multi_scale = self.backbone(x, return_multi_scale=True)
            cls_logits = self.temp_head(feat4)
            seg_pred = self.seg_decoder(multi_scale)
            
            return {
                "logits": cls_logits,
                "seg_pred": seg_pred,
                "concept_scores": None,
                "features": feat4,
                "spatial_sim_map": None
            }
        
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
        
        # ★ intervention 모드: concept_scores를 외부에서 조작할 수 있도록
        # head만 따로 호출 가능하게 함
        elif self.mode == 'intervention':
            with torch.no_grad():
                raw_features = self.backbone(x)
                aligned_features = self.projector(raw_features)
                concept_scores, spatial_sim_map = self.prototypes(aligned_features)
            
            # concept_scores를 반환만 하고 head는 호출하지 않음
            # (intervention.py에서 concept_scores 조작 후 head를 따로 호출)
            return {
                "logits": None,
                "seg_pred": None,
                "concept_scores": concept_scores,
                "features": aligned_features,
                "spatial_sim_map": spatial_sim_map
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def predict_from_concepts(self, concept_scores):
        """
        ★ Intervention용: 조작된 concept_scores로 예측.
        CBM의 Intervenability를 구현하는 핵심 메서드.
        
        Args:
            concept_scores: [B, score_dim] (조작 가능)
        Returns:
            logits: [B, num_classes]
        """
        with torch.no_grad():
            return self.head(concept_scores)

    def set_session_mode(self, mode):
        valid_modes = ['pretrain', 'extract', 'head_train', 'incremental', 'eval', 'intervention']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Valid: {valid_modes}")
        
        print(f"\n[*] DICAN mode → {mode.upper()}")
        self.mode = mode
        
        if mode == 'pretrain':
            self.backbone.set_session_mode('base')
            for p in self.temp_head.parameters():
                p.requires_grad = True
            self.temp_head.train()
            for p in self.seg_decoder.parameters():
                p.requires_grad = True
            self.seg_decoder.train()
            for p in self.projector.parameters():
                p.requires_grad = False
            self.projector.eval()
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()
            
            print("   → Backbone:    Trainable")
            print("   → TempHead:    Trainable")
            print("   → SegDecoder:  Trainable (224×224)")
            print(f"   → Projector({self.projector_type})/CBM Head: Frozen")
            
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
            
            print("   → Backbone:  Frozen")
            print("   → CBM Head:  Trainable (Enhanced 12→64→32→16→5)")
            print("   → logit_scale: Trainable")
            
        elif mode == 'incremental':
            self.backbone.set_session_mode('incremental')
            self.projector.set_session_mode('incremental')
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()
            self.prototypes.logit_scale.requires_grad = False
            print(f"   → Projector({self.projector_type}): Trainable "
                  f"({self.projector.get_param_count():,} params)")
            print("   → All else:  Frozen")
            
        elif mode in ('eval', 'intervention'):
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

    def set_eval_mode(self, task_id=0):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        
        if task_id == 0:
            self.projector.mode = 'base'
        else:
            self.projector.mode = 'incremental'