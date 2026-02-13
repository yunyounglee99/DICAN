import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeBank(nn.Module):
    """
    DICAN 모델의 심장부: Prototype 관리 및 연산 모듈.
    
    [역할]
    1. Base Session: Masked Global Average Pooling을 통해 순도 높은 프로토타입을 계산하고 업데이트함.
    2. Incremental Session: 입력 특징($z$)과 저장된 프로토타입($P$) 간의 유사도(Concept Score)를 계산함.
    """
    def __init__(self, num_concepts=4, feature_dim=2048, momentum=0.9):
        """
        Args:
            num_concepts (int): Concept 개수 (DDR: 4 - EX, HE, MA, SE)
            feature_dim (int): Backbone/Projector 출력 차원 (ResNet50: 2048)
            momentum (float): 배치 단위 업데이트 시 이동 평균(EMA) 적용 비율
        """
        super(PrototypeBank, self).__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        self.momentum = momentum

        # 프로토타입 저장소 (학습 파라미터가 아님 -> register_buffer)
        # shape: [4, 2048]
        self.register_buffer("prototypes", torch.zeros(num_concepts, feature_dim))
        
        # 초기화 여부 확인용
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    def forward(self, features):
        """
        [Incremental Session용]
        입력 특징과 프로토타입 간의 유사도(Concept Score) 계산
        
        Args:
            features: [Batch, 2048, 7, 7] (Projector를 통과한 특징)
        Returns:
            similarity: [Batch, num_concepts] (0~1 사이 값)
        """
        # 1. Spatial Pooling (공간 차원 압축)
        # Projector를 통과한 특징은 위치 정보가 정렬되었으므로 GAP 수행
        z = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1) # [Batch, 2048]
        
        # 2. 정규화 (Cosine Similarity를 위해)
        z_norm = F.normalize(z, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # 3. 유사도 계산 (Dot Product)
        # [Batch, 2048] @ [2048, 4] -> [Batch, 4]
        similarity = torch.mm(z_norm, p_norm.t())
        
        return similarity

    def update_with_masks(self, features, masks):
        """
        [Base Session용 - 핵심 함수]
        마스크를 사용하여 '진짜 병변'의 특징만 추출해 프로토타입을 업데이트함.
        학습 Loop에서 배치마다 호출됨.
        
        Args:
            features: [Batch, 2048, 7, 7] (Backbone 출력)
            masks: [Batch, 4, 224, 224] (원본 마스크) - 순서는 Concept 순서와 같아야 함
        Returns:
            concept_vectors: [Batch, 4] (현재 배치의 컨셉 벡터 - Head 학습용)
        """
        batch_size = features.size(0)
        feat_h, feat_w = features.shape[2], features.shape[3] # 7, 7
        
        # 1. 마스크 리사이징 (224x224 -> 7x7)
        # 정보 손실 방지를 위해 'nearest' 모드나 'max pooling' 추천
        # 여기서는 간단히 interpolate nearest 사용
        masks_resized = F.interpolate(masks, size=(feat_h, feat_w), mode='nearest') # [B, 4, 7, 7]
        
        current_batch_prototypes = []
        
        # 2. 각 Concept 별로 Masked Average Pooling 수행
        for k in range(self.num_concepts):
            # k번째 Concept의 마스크: [B, 1, 7, 7]
            mask_k = masks_resized[:, k:k+1, :, :] 
            
            # 마스크가 하나라도 켜진(1) 픽셀이 있는지 확인
            mask_sum = mask_k.sum(dim=(2, 3), keepdim=True) # [B, 1, 1, 1]
            mask_sum = torch.clamp(mask_sum, min=1e-6) # 0으로 나누기 방지
            
            # Masked Mean 계산: (Feature * Mask) / Mask_Sum
            # Feature: [B, 2048, 7, 7], Mask: [B, 1, 7, 7] -> Broadcasting
            masked_feat = (features * mask_k).sum(dim=(2, 3), keepdim=True) / mask_sum
            masked_feat = masked_feat.view(batch_size, -1) # [B, 2048]
            
            current_batch_prototypes.append(masked_feat)

            # 3. Global Prototype 업데이트 (Moving Average)
            # 현재 배치에서 병변이 실제로 존재하는 샘플만 골라서 업데이트
            valid_indices = (mask_k.view(batch_size, -1).sum(dim=1) > 0) # [B]
            if valid_indices.any():
                valid_feats = masked_feat[valid_indices] # [valid_B, 2048]
                new_proto = valid_feats.mean(dim=0) # [2048]
                
                # 이동 평균 업데이트
                if self.initialized[0]:
                    self.prototypes[k] = self.momentum * self.prototypes[k] + \
                                         (1 - self.momentum) * new_proto
                else:
                    self.prototypes[k] = new_proto

        # 초기화 완료 플래그 설정
        self.initialized.fill_(True)
        
        # 현재 배치의 컨셉 벡터 반환 (이걸로 Head를 학습시킴!)
        # [Batch, 4, 2048] -> [Batch, 4] (유사도 형태로 변환하여 Head에 입력)
        # Head 학습을 위해 현재 프로토타입과의 유사도를 리턴
        batch_concepts = torch.stack(current_batch_prototypes, dim=1) # [B, 4, 2048]
        
        # Cosine Similarity 계산하여 리턴 (0~1)
        z_norm = F.normalize(batch_concepts, p=2, dim=2)
        p_norm = F.normalize(self.prototypes, p=2, dim=1) # [4, 2048]
        
        # 각 샘플의 k번째 컨셉 벡터와 k번째 프로토타입 간의 유사도 (Diagonal 성분)
        # einsum을 쓰면 편함: [B, K, D] * [K, D] -> [B, K]
        concept_scores = torch.einsum('bkd,kd->bk', z_norm, p_norm)
        
        return concept_scores

    def save_prototypes(self, path):
        torch.save(self.prototypes.cpu(), path)
        print(f"Prototypes saved to {path}")

    def load_prototypes(self, path):
        self.prototypes = torch.load(path).to(self.prototypes.device)
        self.initialized.fill_(True)
        print(f"Prototypes loaded from {path}")