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
        labels = targets['label']         # [B]
        logits = outputs['logits']        # [B, 5]
        concept_scores = outputs['concept_scores'] # [B, 4] (Cosine Sim)
        
        # 1. Alignment Loss (Max Pooling for Sub-concepts)
        # [B, 20] -> [B, 4, 5] -> [B, 4] (각 컨셉별 최댓값 추출)
        batch_size = concept_scores.size(0)
        scores_reshaped = concept_scores.view(batch_size, self.num_concepts, -1) # -1 is num_sub_concepts
        
        # Max Pooling: 서브 컨셉 중 가장 유사한 것 하나를 대표값으로 사용
        concept_max_scores, _ = torch.max(scores_reshaped, dim=2) 
        
        # Sigmoid로 0~1 변환
        concept_probs = torch.sigmoid(concept_max_scores)
        
        expected_concepts = self.concept_rule_matrix[labels] # [B, 4]
        loss_align = F.binary_cross_entropy(concept_probs, expected_concepts)

        # 2. Ordinal Loss (Class Weight 적용 추천!)
        # 여기는 Head가 20개 입력을 다 받아서 처리했으므로 logits 그대로 사용
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        
        # 예측된 클래스 (argmax)
        pred_labels = torch.argmax(logits, dim=1)
        
        # Penalty Weight 계산:
        # 실제(Label)보다 예측(Pred)이 낮으면(Under-diagnosis) 페널티 부여
        # 예: 실제 3, 예측 0 -> 차이 3 -> 가중치 크게
        distance = labels - pred_labels
        # distance > 0 인 경우(미진단)에만 가중치 적용 (예: 1 + 차이)
        penalty_weights = 1.0 + torch.relu(distance.float()) * 0.5 
        
        loss_ordinal = (ce_loss * penalty_weights).mean()
        
        # ---------------------------------------------------------
        # Loss 3: Sparsity Constraint (For Normal Images)
        # ---------------------------------------------------------
        # 라벨이 0(Normal)인 샘플만 골라냄
        normal_indices = (labels == 0)
        loss_sparsity = torch.tensor(0.0, device=logits.device)
        
        if normal_indices.any():
            # 정상 이미지의 모든 Concept Score의 절대값 합을 최소화
            # (정상이면 어떤 병변 프로토타입과도 유사하면 안 됨)
            loss_sparsity = torch.mean(torch.abs(concept_scores[normal_indices]))

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