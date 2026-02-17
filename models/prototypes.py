import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeBank(nn.Module):
    """
    DICAN Prototype Bank - Phase 분리 버전
    
    [핵심 변경사항]
    1. Hybrid Pooling (GAP + GMP): 존재 감지(Max) + 확산 정도(Mean)를 동시 포착
    2. Phase 분리: Backbone 학습과 Prototype 구축을 완전히 분리
    3. Scale Factor 복원: Concept Score의 변별력 확보
    
    [Concept Score 차원]
    - 기존: [B, num_concepts]  (4차원 → 정보 부족)
    - 변경: [B, num_concepts * 2] (8차원 → Max + Mean 각각 4개)
    """
    def __init__(self, num_concepts=4, feature_dim=2048, momentum=0.9):
        super(PrototypeBank, self).__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        self.momentum = momentum

        # Prototype 저장소: [K, C] = [4, 2048]
        self.register_buffer("prototypes", torch.zeros(num_concepts, feature_dim))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        
        # ---------------------------------------------------------------
        # [Fix #1] Scale Factor 복원 (학습 가능)
        # Cosine Similarity [-1,1]을 유의미한 logit 범위로 증폭
        # 초기값 ln(20) ≈ 3.0 → exp(3.0) = 20.0배 증폭
        # ---------------------------------------------------------------
        self.logit_scale = nn.Parameter(torch.ones([]) * 3.0)
        
        # Concept Score 출력 차원 (Head에 전달할 정보)
        # Hybrid Pooling: Max + Mean = 2 channels per concept
        self.score_dim = num_concepts * 2

    def get_score_dim(self):
        """Head 초기화 시 입력 차원을 알려주는 함수"""
        return self.score_dim

    def forward(self, features):
        """
        [Hybrid Spatial Concept Matching]
        
        Args:
            features: [B, 2048, 7, 7] (Projector 통과 후 또는 raw features)
        Returns:
            concept_scores: [B, num_concepts * 2] 
                            (Max 4개 + Mean 4개 = 8차원)
            spatial_sim_map: [B, num_concepts, 7, 7] (시각화/Loss용)
        """
        batch, ch, h, w = features.shape
        
        # 1. Feature & Prototype 정규화
        z = F.normalize(features, p=2, dim=1)   # [B, 2048, 7, 7]
        p = F.normalize(self.prototypes, p=2, dim=1)  # [K, 2048]
        
        # 2. 픽셀별 Cosine Similarity
        # [B, C, H, W] × [K, C] → [B, K, H, W]
        spatial_sim_map = torch.einsum('bchw,kc->bkhw', z, p)
        
        # 3. Scale Factor 적용 (★ 핵심 Fix)
        scale = self.logit_scale.exp().clamp(max=100.0)
        scaled_sim = spatial_sim_map * scale
        
        # 4. Hybrid Pooling: Max + Mean 동시 추출
        flat_sim = scaled_sim.flatten(2)  # [B, K, H*W]
        
        # (a) GMP: "가장 강한 병변 신호" → 존재 감지
        #     작은 MA도 1개 픽셀만 강하면 포착됨
        scores_max, _ = flat_sim.max(dim=2)    # [B, K]
        
        # (b) GAP: "평균 활성화 수준" → 병변 확산 정도
        #     HE가 넓게 퍼져있으면 GAP이 높고, 1점이면 낮음
        scores_mean = flat_sim.mean(dim=2)     # [B, K]
        
        # (c) 결합: [B, K*2] = [B, 8]
        # Head가 두 신호를 독립적으로 학습할 수 있도록 concat
        concept_scores = torch.cat([scores_max, scores_mean], dim=1)
        
        return concept_scores, spatial_sim_map

    def compute_spatial_similarity(self, features):
        """
        Segmentation Loss를 위한 raw spatial similarity 계산
        (Scale Factor 미적용 → BCE Loss와 호환)
        
        Args:
            features: [B, 2048, 7, 7]
        Returns:
            spatial_sim: [B, K, 7, 7] (값 범위: -1 ~ 1)
        """
        f_norm = F.normalize(features, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        spatial_sim = torch.einsum('bchw,kc->bkhw', f_norm, p_norm)
        return spatial_sim

    # =================================================================
    # Phase 1-B: 안정적 Prototype 구축 (Backbone Frozen 상태에서 호출)
    # =================================================================
    def extract_prototypes_from_dataset(self, backbone, dataloader, device):
        """
        [Phase 1-B 전용] 
        학습 완료된 Backbone을 사용해 전체 데이터셋에서 Prototype 추출.
        
        핵심: Backbone이 이미 고정되어 있으므로 feature 분포가 안정적이고,
        한 번의 full pass로 깨끗한 prototype을 구축할 수 있음.
        
        Pooling 전략: Masked GAP (Prototype 구축용)
        - GMP가 아닌 GAP을 쓰는 이유: Prototype은 "대표 벡터"여야 하므로
          극단값보다 중심(centroid)이 안정적임
        - GMP의 장점은 inference 시 forward()의 Hybrid Pooling에서 활용됨
        
        Args:
            backbone: Frozen ResNet-50 backbone
            dataloader: DDR Train DataLoader (image + mask 포함)
            device: cuda/cpu
        """
        print("[Phase 1-B] Extracting Prototypes from frozen backbone...")
        backbone.eval()
        
        # 각 Concept별 feature 누적 저장소
        concept_feat_sum = torch.zeros(self.num_concepts, self.feature_dim, device=device)
        concept_count = torch.zeros(self.num_concepts, device=device)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                images = batch_data['image'].to(device)
                masks = batch_data['masks'].to(device).float()
                
                # Backbone feature 추출
                features = backbone(images)  # [B, 2048, 7, 7]
                feat_h, feat_w = features.shape[2], features.shape[3]
                
                # 마스크 리사이징 (224 → 7, Nearest Neighbor)
                masks_resized = F.interpolate(
                    masks, size=(feat_h, feat_w), mode='nearest'
                )  # [B, 4, 7, 7]
                
                # 각 Concept별 Masked GAP
                for k in range(self.num_concepts):
                    mask_k = masks_resized[:, k:k+1, :, :]  # [B, 1, 7, 7]
                    
                    # 마스크가 있는 픽셀만의 feature 합산
                    masked_feat = (features * mask_k).sum(dim=(2, 3))  # [B, 2048]
                    mask_pixel_count = mask_k.sum(dim=(2, 3)).squeeze()  # [B]
                    
                    # 유효 샘플 (병변이 실제로 있는 이미지만)
                    valid = mask_pixel_count > 0  # [B] boolean
                    
                    if valid.any():
                        # 각 샘플에서 masked average를 구한 뒤 누적
                        valid_feat = masked_feat[valid]  # [N_valid, 2048]
                        valid_count = mask_pixel_count[valid].unsqueeze(1)  # [N_valid, 1]
                        
                        # 샘플별 평균 → 전체 누적
                        per_sample_avg = valid_feat / valid_count  # [N_valid, 2048]
                        concept_feat_sum[k] += per_sample_avg.sum(dim=0)
                        concept_count[k] += valid.sum().float()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"    Processed {batch_idx + 1} batches...")
        
        # 최종 Prototype 계산: 전체 평균 → 정규화
        for k in range(self.num_concepts):
            if concept_count[k] > 0:
                self.prototypes[k] = concept_feat_sum[k] / concept_count[k]
                self.prototypes[k] = F.normalize(self.prototypes[k], p=2, dim=0)
                print(f"    Concept {k}: {int(concept_count[k].item())} samples used")
            else:
                print(f"    [Warning] Concept {k}: No valid samples found!")
        
        self.initialized.fill_(True)
        
        # Prototype 간 유사도 출력 (디버깅용)
        self._print_prototype_similarity()
    
    def _print_prototype_similarity(self):
        """Prototype 간 Cosine Similarity 출력 (분리도 확인)"""
        p = F.normalize(self.prototypes, p=2, dim=1)
        sim_matrix = torch.mm(p, p.t())
        concept_names = ["EX", "HE", "MA", "SE"]
        
        print("\n    [Prototype Similarity Matrix]")
        header = "         " + "  ".join(f"{n:>6}" for n in concept_names)
        print(header)
        for i, name in enumerate(concept_names):
            row = f"    {name:>4} " + "  ".join(f"{sim_matrix[i,j].item():>6.3f}" for j in range(4))
            print(row)
        print()

    # =================================================================
    # Phase 1-C: Head 학습용 (Backbone + Prototype 모두 Frozen)
    # =================================================================
    def compute_scores_for_head_training(self, features):
        """
        Phase 1-C에서 Head를 학습시킬 때 사용.
        forward()와 동일하지만 명시적으로 detach하여 
        gradient가 Backbone/Prototype으로 전파되지 않음을 보장.
        """
        return self.forward(features.detach())

    # =================================================================
    # Base Session 호환 (기존 update_with_masks 대체)
    # =================================================================
    def update_with_masks(self, features, masks, update_prototype=True):
        """
        [하위 호환성 유지]
        Phase 분리 구조에서는 이 함수가 Phase 1-B에서만 호출됨.
        Phase 1-A에서는 호출되지 않음 (순환 의존성 제거).
        
        Returns:
            concept_scores: [B, num_concepts * 2] (Hybrid Pooling)
        """
        feat_h, feat_w = features.shape[2], features.shape[3]
        masks_resized = F.interpolate(masks, size=(feat_h, feat_w), mode='nearest')
        batch_size = features.size(0)
        
        if update_prototype:
            with torch.no_grad():
                for k in range(self.num_concepts):
                    mask_k = masks_resized[:, k:k+1, :, :]
                    mask_sum = mask_k.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                    masked_feat = (features * mask_k).sum(dim=(2, 3), keepdim=True) / mask_sum
                    
                    valid_indices = (mask_k.view(batch_size, -1).sum(dim=1) > 0)
                    if valid_indices.any():
                        valid_feats = masked_feat[valid_indices].view(-1, self.feature_dim)
                        new_proto = valid_feats.mean(dim=0)
                        new_proto = F.normalize(new_proto, p=2, dim=0)
                        
                        if self.initialized[0]:
                            self.prototypes[k] = (self.momentum * self.prototypes[k] + 
                                                  (1 - self.momentum) * new_proto)
                        else:
                            self.prototypes[k] = new_proto
                
                self.initialized.fill_(True)
        
        # Head 학습용 점수 반환 (Hybrid Pooling 적용)
        concept_scores, _ = self.forward(features)
        return concept_scores

    def save_prototypes(self, path):
        """Prototype 벡터 저장"""
        save_dict = {
            'prototypes': self.prototypes.cpu(),
            'logit_scale': self.logit_scale.data.cpu(),
            'initialized': self.initialized.cpu()
        }
        torch.save(save_dict, path)
        print(f"[*] Prototypes saved to {path}")
        print(f"    Scale Factor: {self.logit_scale.exp().item():.2f}")

    def load_prototypes(self, path):
        """Prototype 벡터 로드"""
        checkpoint = torch.load(path, map_location=self.prototypes.device)
        
        if isinstance(checkpoint, dict) and 'prototypes' in checkpoint:
            self.prototypes.copy_(checkpoint['prototypes'].to(self.prototypes.device))
            if 'logit_scale' in checkpoint:
                self.logit_scale.data.copy_(checkpoint['logit_scale'].to(self.prototypes.device))
        else:
            # 하위 호환: 이전 형식 (tensor만 저장된 경우)
            self.prototypes.copy_(checkpoint.to(self.prototypes.device))
            
        self.initialized.fill_(True)
        print(f"[*] Prototypes loaded from {path}")


# --- 테스트 코드 ---
if __name__ == "__main__":
    print("=== PrototypeBank Hybrid Pooling Test ===\n")
    
    bank = PrototypeBank(num_concepts=4, feature_dim=2048)
    
    # 가짜 Prototype 설정 (정규화된 랜덤 벡터)
    fake_protos = torch.randn(4, 2048)
    fake_protos = F.normalize(fake_protos, p=2, dim=1)
    bank.prototypes.copy_(fake_protos)
    bank.initialized.fill_(True)
    
    # 더미 Feature
    dummy_feat = torch.randn(2, 2048, 7, 7)
    
    # Forward
    scores, sim_map = bank(dummy_feat)
    
    print(f"Concept Scores shape: {scores.shape}")  # [2, 8]
    print(f"  -> Max scores (첫 4개): {scores[0, :4].tolist()}")
    print(f"  -> Mean scores (뒤 4개): {scores[0, 4:].tolist()}")
    print(f"Spatial Sim Map shape: {sim_map.shape}")  # [2, 4, 7, 7]
    print(f"Scale Factor: {bank.logit_scale.exp().item():.2f}")
    print(f"Score Dim for Head: {bank.get_score_dim()}")  # 8
    
    # Max와 Mean의 차이 확인
    diff = (scores[:, :4] - scores[:, 4:]).abs().mean()
    print(f"\nMax-Mean difference (avg): {diff.item():.4f}")
    print("  -> 차이가 클수록 Hybrid Pooling의 의미가 있음")