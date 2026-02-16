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
    
    def compute_spatial_similarity(self, features):
        """
        [Fix] Base Session의 Segmentation Loss를 위한 공간 유사도 맵 계산
        Args:
            features: [Batch, 2048, 7, 7]
        Returns:
            spatial_sim: [Batch, num_concepts, 7, 7]
        """
        # Feature 정규화 (Channel 차원 기준)
        f_norm = F.normalize(features, p=2, dim=1) # [B, 2048, 7, 7]
        # Prototype 정규화
        p_norm = F.normalize(self.prototypes, p=2, dim=1) # [K, 2048]
        
        # 1x1 Convolution 처럼 연산: (B, C, H, W) * (K, C) -> (B, K, H, W)
        # einsum 사용: bchw, kc -> bkhw
        spatial_sim = torch.einsum('bchw,kc->bkhw', f_norm, p_norm)
        return spatial_sim

    def update_with_masks(self, features, masks, update_prototype=True):
        """
        Args:
            update_prototype (bool): True면 EMA로 프로토타입 갱신 (Train용), 
                                     False면 값만 계산 (Val용) - [Fix]
        """
        batch_size = features.size(0)
        feat_h, feat_w = features.shape[2], features.shape[3]
        
        masks_resized = F.interpolate(masks, size=(feat_h, feat_w), mode='nearest')
        
        current_batch_prototypes = []
        
        for k in range(self.num_concepts):
            mask_k = masks_resized[:, k:k+1, :, :] 
            mask_sum = mask_k.sum(dim=(2, 3), keepdim=True)
            mask_sum = torch.clamp(mask_sum, min=1e-6)
            
            masked_feat = (features * mask_k).sum(dim=(2, 3), keepdim=True) / mask_sum
            masked_feat = masked_feat.view(batch_size, -1)
            current_batch_prototypes.append(masked_feat)

            # [Fix] update_prototype 플래그가 True일 때만 업데이트 수행
            if update_prototype:
                valid_indices = (mask_k.view(batch_size, -1).sum(dim=1) > 0)
                if valid_indices.any():
                    valid_feats = masked_feat[valid_indices].detach()
                    new_proto = valid_feats.mean(dim=0)
                    
                    if self.initialized[0]:
                        self.prototypes[k] = self.momentum * self.prototypes[k] + \
                                             (1 - self.momentum) * new_proto
                    else:
                        self.prototypes[k] = new_proto
        
        if update_prototype:
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