import torch
import torch.nn as nn
import torch.nn.functional as F

class DICANLoss(nn.Module):
    """
    DICAN 모델을 위한 통합 손실 함수 모듈.
    Base Session과 Incremental Session의 서로 다른 학습 목표를 모두 지원함.
    """
    def __init__(self, mode='base', num_concepts=4, num_classes=5):
        super(DICANLoss, self).__init__()
        self.mode = mode
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # 가중치 설정 (실험적으로 튜닝 가능)
        self.lambda_ordinal = 1.0
        self.lambda_sparsity = 0.5
        self.lambda_seg = 1.0

        # [Alignment Logic] DR 등급별로 반드시 존재해야 하는 병변 정의 (Medical Prior)
        # 예: DDR 데이터셋 기준 (0:None, 1:MA, 2:MA+HE+EX, 3:SE..., 4:ALL)
        # 이 매트릭스는 "Grade Y라면 Concept C가 켜져야 한다(1)"를 정의함.
        # Row: Grade(0~4), Col: Concept(EX, HE, MA, SE)
        # *주의: 실제 데이터셋 통계에 맞춰 수정 필요
        self.register_buffer('concept_rule_matrix', torch.tensor([
            [0, 0, 0, 0], # Grade 0: 정상 (아무것도 없음)
            [0, 0, 1, 0], # Grade 1: MA 위주
            [1, 1, 1, 0], # Grade 2: EX, HE, MA
            [1, 1, 1, 1], # Grade 3: SE 추가
            [1, 1, 1, 1]  # Grade 4: 심각 (모두 존재 가능)
        ], dtype=torch.float32))

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): 모델의 출력 
                            {'logits': ..., 'concept_scores': ..., 'features': ...}
            targets (dict): 정답 데이터 
                            {'label': ..., 'masks': ... (Base only)}
        """
        if self.mode == 'base':
            return self._forward_base(outputs, targets)
        elif self.mode == 'incremental':
            return self._forward_incremental(outputs, targets)
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")

    # =========================================================================
    # 1. Base Session Loss Logic
    # =========================================================================
    def _forward_base(self, outputs, targets):
        labels = targets['label']
        logits = outputs['logits']
        masks = targets['masks'] # [B, 4, 224, 224]
        
        # A. Classification Loss (Class Weight 적용)
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
        
        # B. Segmentation Loss (Concept-Specific) [Fix for Sub-Concepts]
        if outputs['spatial_sim_map'] is not None:
            spatial_sim = outputs['spatial_sim_map'] # 현재 shape: [B, 20, 7, 7]
            
            # 1. 마스크 리사이징
            masks_resized = F.interpolate(masks, size=spatial_sim.shape[2:], mode='nearest') # [B, 4, 7, 7]
            
            # 2. [New] Sub-Concept Aggregation (20ch -> 4ch)
            # "출혈"에 해당하는 5개 서브 프로토타입 중 가장 유사도가 높은 것을 대표값으로 사용
            batch, total_k, h, w = spatial_sim.shape
            num_sub = total_k // self.num_concepts # 20 // 4 = 5
            
            # [B, 4, 5, 7, 7] 형태로 변환
            sim_reshaped = spatial_sim.view(batch, self.num_concepts, num_sub, h, w)
            
            # Max Pooling over Sub-concepts (dim=2)
            # 결과: [B, 4, 7, 7]
            spatial_sim_aggregated, _ = torch.max(sim_reshaped, dim=2)
            
            # 3. 확률 변환 (Cosine Sim -1~1 -> Prob 0~1)
            # 주의: Scale Factor가 이미 곱해져 있다면 Sigmoid를 써야 하고, 
            # 순수 Cosine Sim이라면 Linear Mapping을 해야 합니다.
            # PrototypeBank의 compute_spatial_similarity는 아직 Scaling 전의 raw cosine 값을 줍니다.
            # 따라서 (sim + 1) * 0.5 가 안전합니다.
            spatial_probs = (spatial_sim_aggregated + 1) * 0.5
            
            # 4. Loss 계산
            loss_seg = F.binary_cross_entropy(spatial_probs, masks_resized)
        else:
            loss_seg = torch.tensor(0.0, device=logits.device)

        total_loss = loss_cls + self.lambda_seg * loss_seg
        return total_loss, {"loss_cls": loss_cls.item(), "loss_seg": loss_seg.item()}

    # =========================================================================
    # 2. Incremental Session Loss Logic (The Core of DICAN)
    # =========================================================================
    def _forward_incremental(self, outputs, targets):
        labels = targets['label']
        logits = outputs['logits']
        
        # [Fix] 기존 concept_scores 대신 공간 맵(Spatial Map)을 사용합니다.
        # Shape: [Batch, 4, 7, 7] (4는 Concept 개수)
        # dican_cbm.py의 forward에서 이 값을 리턴하도록 수정했는지 꼭 확인하세요.
        spatial_sim_map = outputs['spatial_sim_map'] 
        
        # ---------------------------------------------------------
        # 1. Alignment Loss (Spatial Max Pooling)
        # ---------------------------------------------------------
        # "이미지 49개 구역(7x7) 중, 단 한 곳이라도 병변 특징이 강하면 '있다'고 판단한다"
        # [B, 4, 7, 7] -> Flatten [B, 4, 49] -> Max [B, 4]
        concept_max_scores, _ = spatial_sim_map.flatten(2).max(dim=2)
        
        # Sigmoid로 확률 변환 (0~1)
        # (PrototypeBank에서 이미 Scale Factor가 곱해져서 넘어옴)
        concept_probs = torch.sigmoid(concept_max_scores)
        
        # 정답지 (Grade별 규칙) 가져오기
        expected_concepts = self.concept_rule_matrix[labels] # [B, 4]
        
        # BCE Loss 계산
        loss_align = F.binary_cross_entropy(concept_probs, expected_concepts)

        # ---------------------------------------------------------
        # 2. Ordinal Regression Loss (Class Weight 적용)
        # ---------------------------------------------------------
        # 데이터 불균형 해결을 위한 가중치 (Grade 0은 적게, 나머지는 크게)
        
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        pred_labels = torch.argmax(logits, dim=1)
        # 미진단(Under-diagnosis)에만 페널티 부여
        distance = labels - pred_labels
        penalty_weights = 1.0 + torch.relu(distance.float()) * 0.5 
        
        loss_ordinal = (ce_loss * penalty_weights).mean()
        
        # ---------------------------------------------------------
        # 3. Sparsity Constraint (정상 이미지 억제)
        # ---------------------------------------------------------
        normal_indices = (labels == 0)
        loss_sparsity = torch.tensor(0.0, device=logits.device)
        
        if normal_indices.any():
            # 정상 이미지(Grade 0)라면, 7x7 공간 어디에서도 반응이 없어야 함
            # 맵 전체의 활성도(Probability) 평균을 낮춤
            normal_probs = torch.sigmoid(spatial_sim_map[normal_indices])
            loss_sparsity = normal_probs.mean()

        # Total Loss
        total_loss = loss_align + \
                     self.lambda_ordinal * loss_ordinal + \
                     self.lambda_sparsity * loss_sparsity
                    
        return total_loss, {
            "loss_align": loss_align.item(),
            "loss_ordinal": loss_ordinal.item(),
            "loss_sparsity": loss_sparsity.item()
        }

    def set_mode(self, mode):
        self.mode = mode