import torch
import torch.nn as nn
import torch.nn.functional as F

class DICANLoss(nn.Module):
    """
    DICAN Loss - Hybrid Pooling 대응 버전
    
    [변경사항]
    - concept_scores가 [B, 8] (Max 4개 + Mean 4개)
    - Alignment Loss: Max scores를 기준으로 병변 존재 여부 판단
    - Sparsity Loss: 정상 이미지에서 Max scores 억제
    """
    def __init__(self, mode='base', num_concepts=4, num_classes=5):
        super(DICANLoss, self).__init__()
        self.mode = mode
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        self.lambda_ordinal = 1.0
        self.lambda_sparsity = 0.5
        self.lambda_align = 1.0

        # 의학적 규칙: Grade별 병변 존재 여부
        # Row: Grade(0~4), Col: Concept(EX, HE, MA, SE)
        self.register_buffer('concept_rule_matrix', torch.tensor([
            [0, 0, 0, 0],  # Grade 0: Normal
            [0, 0, 1, 0],  # Grade 1: MA
            [1, 1, 1, 0],  # Grade 2: EX, HE, MA
            [1, 1, 1, 1],  # Grade 3: + SE
            [1, 1, 1, 1]   # Grade 4: All
        ], dtype=torch.float32))

    def forward(self, outputs, targets):
        if self.mode == 'base':
            return self._forward_base(outputs, targets)
        elif self.mode == 'incremental':
            return self._forward_incremental(outputs, targets)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # =========================================================================
    # Base Session: Phase 1-C에서 사용 (Head 학습)
    # =========================================================================
    def _forward_base(self, outputs, targets):
        labels = targets['label']
        logits = outputs['logits']
        
        # Classification Loss
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
        
        total_loss = loss_cls
        log_dict = {"loss_cls": loss_cls.item()}
        
        # Concept Score 기반 추가 Loss (있는 경우)
        concept_scores = outputs.get('concept_scores')
        if concept_scores is not None:
            # Max scores (처음 num_concepts개)
            max_scores = concept_scores[:, :self.num_concepts]
            
            # Sparsity: Grade 0에서 max scores 억제
            normal_mask = (labels == 0)
            if normal_mask.any():
                loss_sparsity = torch.relu(max_scores[normal_mask]).mean()
                total_loss = total_loss + 0.3 * loss_sparsity
                log_dict["loss_sparsity"] = loss_sparsity.item()
        
        return total_loss, log_dict

    # =========================================================================
    # Incremental Session: Projector 학습
    # =========================================================================
    def _forward_incremental(self, outputs, targets):
        labels = targets['label']
        logits = outputs['logits']
        concept_scores = outputs['concept_scores']  # [B, 8]
        
        # Max scores와 Mean scores 분리
        max_scores = concept_scores[:, :self.num_concepts]   # [B, 4]
        mean_scores = concept_scores[:, self.num_concepts:]  # [B, 4]
        
        # ---------------------------------------------------------
        # 1. Alignment Loss
        # Max scores → Sigmoid → BCE with medical rules
        # ---------------------------------------------------------
        concept_probs = torch.sigmoid(max_scores)
        expected_concepts = self.concept_rule_matrix[labels]  # [B, 4]
        loss_align = F.binary_cross_entropy(concept_probs, expected_concepts)

        # ---------------------------------------------------------
        # 2. Ordinal Regression Loss
        # ---------------------------------------------------------
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        
        pred_labels = torch.argmax(logits, dim=1)
        distance = labels - pred_labels
        penalty_weights = 1.0 + torch.relu(distance.float()) * 0.5
        loss_ordinal = (ce_loss * penalty_weights).mean()
        
        # ---------------------------------------------------------
        # 3. Sparsity Constraint
        # ---------------------------------------------------------
        normal_indices = (labels == 0)
        loss_sparsity = torch.tensor(0.0, device=logits.device)
        if normal_indices.any():
            # Max + Mean 모두 억제
            normal_all_scores = concept_scores[normal_indices]
            loss_sparsity = torch.relu(normal_all_scores).mean()

        # Total
        total_loss = (self.lambda_align * loss_align + 
                     self.lambda_ordinal * loss_ordinal + 
                     self.lambda_sparsity * loss_sparsity)
                    
        return total_loss, {
            "loss_align": loss_align.item(),
            "loss_ordinal": loss_ordinal.item(),
            "loss_sparsity": loss_sparsity.item()
        }

    def set_mode(self, mode):
        self.mode = mode