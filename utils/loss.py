import torch
import torch.nn as nn
import torch.nn.functional as F

class DICANLoss(nn.Module):
    """
    DICAN Loss - Multi-Cluster Prototype 대응 버전
    
    [변경사항]
    - concept_scores: [B, 12] (Max 4개 + Mean 4개 + Std 4개)
    - Alignment Loss: Max scores 기준 (병변 존재 여부)
    - Sparsity Loss: Max + Mean 동시 억제 (정상 이미지)
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

    def _forward_base(self, outputs, targets):
        labels = targets['label']
        logits = outputs['logits']
        
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
        
        total_loss = loss_cls
        log_dict = {"loss_cls": loss_cls.item()}
        
        concept_scores = outputs.get('concept_scores')
        if concept_scores is not None:
            max_scores = concept_scores[:, :self.num_concepts]
            normal_mask = (labels == 0)
            if normal_mask.any():
                loss_sparsity = torch.relu(max_scores[normal_mask]).mean()
                total_loss = total_loss + 0.3 * loss_sparsity
                log_dict["loss_sparsity"] = loss_sparsity.item()
        
        return total_loss, log_dict

    def _forward_incremental(self, outputs, targets):
        labels = targets['label']
        logits = outputs['logits']
        concept_scores = outputs['concept_scores']  # [B, 12]
        
        nc = self.num_concepts
        max_scores = concept_scores[:, :nc]           # [B, 4]
        mean_scores = concept_scores[:, nc:2*nc]      # [B, 4]
        # std_scores = concept_scores[:, 2*nc:]       # [B, 4] (alignment에는 미사용)
        
        # 1. Alignment Loss (Max scores 기반)
        concept_probs = torch.sigmoid(max_scores)
        expected_concepts = self.concept_rule_matrix[labels]
        loss_align = F.binary_cross_entropy(concept_probs, expected_concepts)

        # 2. Ordinal Regression Loss
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=logits.device)
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        
        pred_labels = torch.argmax(logits, dim=1)
        distance = labels - pred_labels
        penalty_weights = 1.0 + torch.relu(distance.float()) * 0.5
        loss_ordinal = (ce_loss * penalty_weights).mean()
        
        # 3. Sparsity (Max + Mean 동시 억제)
        normal_indices = (labels == 0)
        loss_sparsity = torch.tensor(0.0, device=logits.device)
        if normal_indices.any():
            normal_max = max_scores[normal_indices]
            normal_mean = mean_scores[normal_indices]
            loss_sparsity = (torch.relu(normal_max).mean() + 
                           torch.relu(normal_mean).mean()) * 0.5

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
