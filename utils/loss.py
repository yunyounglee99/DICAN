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
        masks = targets['masks']

        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        
        # A. Classification Loss (Head 학습용)
        loss_cls = F.cross_entropy(logits, labels)
        
        # B. Segmentation Loss (Backbone 학습용)
        # Backbone의 Feature Map과 GT Mask 간의 차이 계산
        # features: [B, 2048, 7, 7] -> Channel-wise mean or 1x1 conv to [B, 4, 7, 7]
        # *주의: Backbone 출력은 2048채널이므로, 마스크(4채널)와 비교하려면
        # 보통 별도의 SegHead가 있거나, 여기서는 간단히 Backbone 학습 유도를 위해
        # 'Attention Map'과 마스크를 비교하는 방식을 사용.
        # (DICAN 논리상 Masked GAP를 쓰므로 Explicit Seg Loss는 보조적 역할임)
        
        # 여기서는 구현의 단순화를 위해, Backbone Features가 마스크 영역에서
        # 높은 activation을 갖도록 유도하는 방식을 생략하고 Cls Loss에 집중하거나,
        # 만약 모델 내부에 Auxiliary Seg Head가 있다면 그 출력을 사용함.
        # (README 설계상 L_seg는 Backbone 학습용이라고 명시됨)
        
        if outputs['spatial_sim_map'] is not None:
            spatial_sim = outputs['spatial_sim_map'] # [B, 4, 7, 7] (값 범위: -1 ~ 1)
            
            # 마스크 리사이징 (Nearest)
            masks_resized = F.interpolate(masks, size=spatial_sim.shape[2:], mode='nearest') # [B, 4, 7, 7]
            
            # Cosine Similarity(-1~1)를 확률(0~1)로 변환
            # (sim + 1) / 2
            spatial_probs = (spatial_sim + 1) * 0.5
            
            # Pixel-wise BCE Loss 계산
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
        
        # ---------------------------------------------------------
        # Loss 1: Alignment Loss (Weakly Supervised Metric Learning)
        # ---------------------------------------------------------
        # 라벨(Grade)을 보고 "있어야 할 Concept"을 추론
        expected_concepts = self.concept_rule_matrix[labels] # [B, 4]
        
        # Projector가 뱉은 Concept Score가 Expected Concept과 같아지도록 유도
        # (있어야 할 건 1에 가깝게, 없어야 할 건 0에 가깝게)
        # Cosine Similarity는 -1~1 범위지만, PrototypeBank에서 보통 Normalize 등을 거침.
        # 여기서는 0~1 범위로 가정하거나 BCEWithLogits 사용.
        
        # Cosine Sim(-1~1)을 0~1 확률로 매핑 (Calibration)
        concept_probs = (concept_scores + 1) / 2 
        loss_align = F.binary_cross_entropy(concept_probs, expected_concepts)
        
        # ---------------------------------------------------------
        # Loss 2: Ordinal Regression Loss (Under-diagnosis Penalty)
        # ---------------------------------------------------------
        # 기본 CE Loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
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