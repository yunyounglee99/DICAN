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
    def __init__(self, num_concepts=4, feature_dim=2048, momentum=0.9, num_sub_concepts=5):
        """
        Args:
            num_concepts (int): Concept 개수 (DDR: 4 - EX, HE, MA, SE)
            feature_dim (int): Backbone/Projector 출력 차원 (ResNet50: 2048)
            momentum (float): 배치 단위 업데이트 시 이동 평균(EMA) 적용 비율
        """
        super(PrototypeBank, self).__init__()
        self.num_concepts = num_concepts
        self.num_sub_concepts = num_sub_concepts
        self.feature_dim = feature_dim
        self.momentum = momentum

        # 프로토타입 저장소 (학습 파라미터가 아님 -> register_buffer)
        # shape: [4, 2048]
        self.total_prototypes = num_concepts * num_sub_concepts
        self.register_buffer("prototypes", torch.zeros(self.total_prototypes, feature_dim))
        
        # 초기화 여부 확인용
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

        self.scale_factor = 20

    def forward(self, features):
        """
        [Spatial Concept Matching Strategy]
        Args:
            features: [Batch, 2048, 7, 7]
        Returns:
            concept_scores: [Batch, num_concepts] (최종 점수)
            spatial_sim_map: [Batch, num_concepts, 7, 7] (시각화 및 Loss용)
        """
        batch, ch, h, w = features.shape
        
        # 1. Feature 정규화 (채널 기준)
        z = F.normalize(features, p=2, dim=1) # [B, 2048, 7, 7]
        
        # 2. Prototype 정규화
        p = F.normalize(self.prototypes, p=2, dim=1) # [K, 2048]
        
        # 3. 픽셀별 유사도 계산 (1x1 Conv 처럼 동작)
        # [B, C, H, W] * [K, C] -> [B, K, H, W]
        spatial_sim_map = torch.einsum('bchw,kc->bkhw', z, p)
        
        # 4. [Scale Factor 적용] 신호 증폭 (필수!)
        # -1~1 사이 값을 -10~10으로 뻥튀기해야 Head가 학습됨
        spatial_sim_map = spatial_sim_map * self.scale_factor # scale_factor=20.0 추천

        # 5. Spatial Aggregation (Pixel -> Image Level Score)
        # "어디선가 병변이 강하게 떴다면 그것은 병변이다" -> Max Pooling
        # [B, K, 7, 7] -> [B, K]
        concept_scores = F.max_pool2d(spatial_sim_map, kernel_size=(h, w)).view(batch, -1)
        
        return concept_scores, spatial_sim_map
    
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
        [Spatial Concept 전략 적용]
        1. 프로토타입 업데이트: 'Masked Average' (안정적인 기준점 생성)
        2. 점수 반환: 'Spatial Max' (Incremental Session과 동일한 로직으로 Head 학습)
        """
        # features: [B, 2048, 7, 7]
        batch_size = features.size(0)
        feat_h, feat_w = features.shape[2], features.shape[3]
        
        # 마스크 리사이징 (224 -> 7)
        masks_resized = F.interpolate(masks, size=(feat_h, feat_w), mode='nearest')
        
        # ----------------------------------------------------------------------
        # 1. 프로토타입 업데이트 (Reference Update)
        #    - 방식: Masked Average (병변 영역의 중심점을 찾음)
        #    - 이유: 프로토타입은 노이즈 없이 깨끗해야 하므로 평균을 씁니다.
        # ----------------------------------------------------------------------
        if update_prototype:
            for k in range(self.num_concepts):
                mask_k = masks_resized[:, k:k+1, :, :] 
                mask_sum = mask_k.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                
                # Masked Average Feature 계산
                masked_feat = (features * mask_k).sum(dim=(2, 3), keepdim=True) / mask_sum
                
                # 유효한 샘플(병변이 있는 경우)만 골라서 EMA 업데이트
                valid_indices = (mask_k.view(batch_size, -1).sum(dim=1) > 0)
                if valid_indices.any():
                    # .detach() 필수! (Backbone 학습에 영향을 주지 않기 위해)
                    valid_feats = masked_feat[valid_indices].view(-1, self.feature_dim).detach() 
                    new_proto = valid_feats.mean(dim=0)
                    
                    # 프로토타입 정규화 (방향성 유지)
                    new_proto = F.normalize(new_proto, p=2, dim=0)

                    if self.initialized[0]:
                        self.prototypes[k] = self.momentum * self.prototypes[k] + \
                                             (1 - self.momentum) * new_proto
                    else:
                        self.prototypes[k] = new_proto
            
            self.initialized.fill_(True)

        # ----------------------------------------------------------------------
        # 2. [핵심 변경] Head 학습용 점수 반환 (Training-Inference Alignment)
        #    - 기존: 평균 벡터끼리 유사도 계산 -> (X) Inc 모드와 로직 다름
        #    - 변경: forward() 함수 호출 -> (O) Spatial Max & Scaling 적용됨
        # ----------------------------------------------------------------------
        # forward 내부 동작: Spatial Similarity Map 생성 -> Scale(x20) -> Max Pooling
        # 결과: Head는 "가장 강한 병변 신호"를 보고 학습하게 됨 (Incremental 때와 동일)
        
        concept_scores, _ = self.forward(features)
        
        return concept_scores

    def save_prototypes(self, path):
        torch.save(self.prototypes.cpu(), path)
        print(f"Prototypes saved to {path}")

    def load_prototypes(self, path):
        self.prototypes = torch.load(path).to(self.prototypes.device)
        self.initialized.fill_(True)
        print(f"Prototypes loaded from {path}")