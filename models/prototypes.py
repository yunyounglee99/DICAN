import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """
    DICAN Prototype Bank v2 - Multi-Cluster + Orthogonality
    
    [핵심 변경사항 from v1]
    
    1. Multi-Cluster Prototypes (k=3):
       기존: Concept당 1개 prototype → 4개 concept가 거의 동일한 벡터
       변경: Concept당 k개 cluster center → within-concept 변이 포착
       EX: [exudate_small, exudate_large, exudate_diffuse]
       HE: [hemorrhage_dot, hemorrhage_blot, hemorrhage_flame]
       → forward에서 cluster별 similarity 계산 후 max-pool
       → 출력 차원은 기존과 동일 (num_concepts * 2)
    
    2. Aggressive Masked Feature Extraction:
       기존: masked_feat = features * mask → 마스크 외 영역 feature 누출
       변경: features에서 마스크 영역만 추출 후 L2 normalize
             → 순수하게 병변 영역의 feature만 prototype에 반영
    
    3. Post-Extraction Orthogonalization:
       k-means 추출 후 concept center 간 직교성 강제
       → EX↔MA 유사도 0.95 → 0.3~0.5 수준으로 감소 기대
    """
    
    def __init__(self, num_concepts=4, feature_dim=2048, num_clusters=3):
        super(PrototypeBank, self).__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters

        # Prototype 저장소: [K, num_clusters, C]
        # EX: 3개 cluster, HE: 3개 cluster, ...
        self.register_buffer(
            "prototypes",
            torch.zeros(num_concepts, num_clusters, feature_dim)
        )
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        
        # Scale Factor (학습 가능)
        self.logit_scale = nn.Parameter(torch.ones([]) * 3.0)
        
        # Hybrid Pooling: Max + Mean = 2 per concept
        self.score_dim = num_concepts * 2

    def get_score_dim(self):
        return self.score_dim

    # =================================================================
    # Forward: Multi-Cluster Spatial Concept Matching
    # =================================================================
    def forward(self, features):
        """
        Args:
            features: [B, 2048, 7, 7]
        Returns:
            concept_scores: [B, num_concepts * 2] (Max + Mean)
            spatial_sim_map: [B, num_concepts, 7, 7] (max over clusters)
        """
        B, C, H, W = features.shape
        K = self.num_concepts
        nc = self.num_clusters
        
        # 1. Normalize
        z = F.normalize(features, p=2, dim=1)           # [B, C, H, W]
        all_p = self.prototypes.view(K * nc, C)          # [K*nc, C]
        all_p = F.normalize(all_p, p=2, dim=1)
        
        # 2. Pixel-wise cosine similarity with ALL prototypes
        # [B, K*nc, H, W]
        all_sim = torch.einsum('bchw,kc->bkhw', z, all_p)
        
        # 3. Reshape to [B, K, nc, H, W] then max over clusters
        all_sim = all_sim.view(B, K, nc, H, W)
        spatial_sim_map, _ = all_sim.max(dim=2)  # [B, K, H, W]
        
        # 4. Scale
        scale = self.logit_scale.exp().clamp(max=100.0)
        scaled_sim = spatial_sim_map * scale
        
        # 5. Hybrid Pooling
        flat_sim = scaled_sim.flatten(2)                 # [B, K, H*W]
        scores_max, _ = flat_sim.max(dim=2)              # [B, K]
        scores_mean = flat_sim.mean(dim=2)               # [B, K]
        concept_scores = torch.cat([scores_max, scores_mean], dim=1)  # [B, 2K]
        
        return concept_scores, spatial_sim_map

    # =================================================================
    # Phase 1-B: Multi-Cluster Prototype Extraction
    # =================================================================
    def extract_prototypes_from_dataset(self, backbone, dataloader, device):
        """
        3-Step Extraction:
        1. Collect: 각 concept별 masked feature 수집
        2. Cluster: k-means로 concept당 k개 cluster center 생성
        3. Orthogonalize: concept center 간 직교화
        """
        print("[Phase 1-B] Extracting Multi-Cluster Prototypes...")
        print(f"    Clusters per concept: {self.num_clusters}")
        backbone.eval()
        
        # ─── Step 1: Collect masked features per concept ───
        concept_features = {k: [] for k in range(self.num_concepts)}
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                images = batch_data['image'].to(device)
                masks = batch_data['masks'].to(device).float()
                
                features = backbone(images)  # [B, 2048, 7, 7]
                feat_h, feat_w = features.shape[2], features.shape[3]
                
                # 마스크 리사이징 (224 → 7)
                masks_resized = F.interpolate(
                    masks, size=(feat_h, feat_w), mode='nearest'
                )  # [B, 4, 7, 7]
                
                for k in range(self.num_concepts):
                    mask_k = masks_resized[:, k:k+1, :, :]  # [B, 1, 7, 7]
                    mask_sum = mask_k.sum(dim=(2, 3)).squeeze(-1)  # [B]
                    valid = mask_sum > 0
                    
                    if valid.any():
                        # ★ 마스크 영역만 강하게 추출 (zero-out 아닌 select)
                        valid_feat = features[valid]           # [N, 2048, 7, 7]
                        valid_mask = mask_k[valid]             # [N, 1, 7, 7]
                        valid_sum = mask_sum[valid].unsqueeze(1)  # [N, 1]
                        
                        # Masked GAP: 마스크 외 영역 완전 제거
                        masked = valid_feat * valid_mask       # [N, 2048, 7, 7]
                        per_sample = masked.sum(dim=(2, 3)) / valid_sum  # [N, 2048]
                        
                        concept_features[k].append(per_sample.cpu())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"    Processed {batch_idx + 1} batches...")
        
        # ─── Step 2: K-Means Clustering per concept ───
        print("\n    [K-Means Clustering]")
        for k in range(self.num_concepts):
            if not concept_features[k]:
                print(f"    [Warning] Concept {k}: No samples! Using random init.")
                nn.init.normal_(self.prototypes[k])
                self.prototypes[k] = F.normalize(self.prototypes[k], p=2, dim=1)
                continue
            
            all_feats = torch.cat(concept_features[k], dim=0)  # [N, 2048]
            all_feats = F.normalize(all_feats, p=2, dim=1)
            n_samples = all_feats.shape[0]
            
            print(f"    Concept {k}: {n_samples} samples → {self.num_clusters} clusters")
            
            if n_samples >= self.num_clusters:
                centers = self._kmeans(all_feats.to(device), self.num_clusters, n_iter=30)
            else:
                # 샘플 부족: 평균 + 노이즈로 cluster 생성
                mean_feat = all_feats.mean(0).to(device)
                centers = mean_feat.unsqueeze(0).repeat(self.num_clusters, 1)
                noise = torch.randn_like(centers) * 0.05
                centers = F.normalize(centers + noise, p=2, dim=1)
            
            self.prototypes[k] = centers  # [num_clusters, 2048]
        
        # ─── Step 3: Orthogonalize concept centers ───
        self._orthogonalize_concepts(strength=0.5)
        
        self.initialized.fill_(True)
        self._print_prototype_similarity()

    def _kmeans(self, features, k, n_iter=30):
        """
        Cosine K-Means (PyTorch 구현)
        
        표준 K-Means 대신 cosine distance 사용:
        - Feature가 이미 L2 normalized
        - 유사도 = dot product
        - 거리 = 1 - cosine_similarity
        """
        N, D = features.shape
        
        # K-Means++ 초기화
        centers = self._kmeans_plusplus_init(features, k)
        
        for iteration in range(n_iter):
            # Normalize centers
            centers = F.normalize(centers, p=2, dim=1)
            
            # Assignment: cosine similarity
            sim = torch.mm(features, centers.t())  # [N, k]
            assignments = sim.argmax(dim=1)        # [N]
            
            # Update centers
            new_centers = torch.zeros_like(centers)
            for j in range(k):
                mask = (assignments == j)
                if mask.sum() > 0:
                    new_centers[j] = features[mask].mean(dim=0)
                else:
                    # Empty cluster: re-init from random sample
                    new_centers[j] = features[torch.randint(N, (1,))].squeeze()
            
            # 수렴 체크
            shift = (F.normalize(new_centers, p=2, dim=1) - centers).norm()
            centers = new_centers
            
            if shift < 1e-6:
                break
        
        return F.normalize(centers, p=2, dim=1)  # [k, D]

    def _kmeans_plusplus_init(self, features, k):
        """K-Means++ 초기화: 다양한 초기 center 선택"""
        N, D = features.shape
        centers = []
        
        # 첫 번째 center: 랜덤
        idx = torch.randint(N, (1,)).item()
        centers.append(features[idx])
        
        for _ in range(1, k):
            # 기존 center들과의 최소 거리 계산
            center_stack = torch.stack(centers)  # [c, D]
            sim = torch.mm(features, center_stack.t())  # [N, c]
            max_sim, _ = sim.max(dim=1)  # [N]
            
            # 거리가 먼 점을 확률적으로 선택
            dist = 1.0 - max_sim  # cosine distance
            dist = dist.clamp(min=0)
            prob = dist / (dist.sum() + 1e-8)
            idx = torch.multinomial(prob, 1).item()
            centers.append(features[idx])
        
        return torch.stack(centers)  # [k, D]

    def _orthogonalize_concepts(self, strength=0.5):
        """
        Concept Center 간 직교화 (Modified Gram-Schmidt)
        
        [방법]
        1. 각 concept의 cluster 평균 → concept center
        2. Gram-Schmidt로 직교 기저 생성
        3. 원래 center와 직교 기저를 strength 비율로 blend
        4. 각 cluster를 center 이동에 맞춰 조정
        
        strength=0: 변화 없음
        strength=1: 완전 직교화
        """
        print(f"\n    [Orthogonalization] strength={strength}")
        
        # 1. Concept centers (cluster 평균)
        centers = self.prototypes.mean(dim=1)   # [K, D]
        centers = F.normalize(centers, p=2, dim=1)
        
        # 원본 보존
        original_centers = centers.clone()
        
        # 2. Modified Gram-Schmidt
        ortho = torch.zeros_like(centers)
        for i in range(self.num_concepts):
            v = centers[i].clone()
            for j in range(i):
                # v에서 이미 선택된 기저 방향 제거
                proj = torch.dot(v, ortho[j])
                v = v - proj * ortho[j]
            ortho[i] = F.normalize(v, p=2, dim=0)
        
        # 3. Blend: original ↔ orthogonal
        new_centers = F.normalize(
            (1.0 - strength) * original_centers + strength * ortho,
            p=2, dim=1
        )
        
        # 4. Cluster 조정: center 이동만큼 각 cluster도 이동
        for k in range(self.num_concepts):
            shift = new_centers[k] - original_centers[k]
            for c in range(self.num_clusters):
                self.prototypes[k, c] = F.normalize(
                    self.prototypes[k, c] + shift, p=2, dim=0
                )
        
        # 결과 확인
        after_sim = self._compute_center_similarity()
        print(f"    After orthogonalization:")
        for i in range(self.num_concepts):
            for j in range(i+1, self.num_concepts):
                print(f"      {['EX','HE','MA','SE'][i]}↔{['EX','HE','MA','SE'][j]}: "
                      f"{after_sim[i,j].item():.3f}")

    def _compute_center_similarity(self):
        """Concept center 간 cosine similarity matrix"""
        centers = self.prototypes.mean(dim=1)  # [K, D]
        centers = F.normalize(centers, p=2, dim=1)
        return torch.mm(centers, centers.t())

    def get_orthogonality_loss(self):
        """
        학습 중 정규화용 직교성 손실
        
        Phase 1-C에서 logit_scale 학습 시 추가 가능:
        loss_orth = ||P_centers @ P_centers.T - I||_F^2
        """
        centers = self.prototypes.mean(dim=1)  # [K, D]
        centers = F.normalize(centers, p=2, dim=1)
        gram = torch.mm(centers, centers.t())  # [K, K]
        identity = torch.eye(self.num_concepts, device=gram.device)
        return (gram - identity).pow(2).sum()

    # =================================================================
    # Printing & Diagnostics
    # =================================================================
    def _print_prototype_similarity(self):
        """Prototype 간 Cosine Similarity 출력"""
        concept_names = ["EX", "HE", "MA", "SE"]
        
        # Concept center 간 유사도
        centers = self.prototypes.mean(dim=1)  # [K, D]
        p = F.normalize(centers, p=2, dim=1)
        sim_matrix = torch.mm(p, p.t())
        
        print("\n    [Concept Center Similarity Matrix]")
        header = "         " + "  ".join(f"{n:>6}" for n in concept_names)
        print(header)
        for i, name in enumerate(concept_names):
            row = f"    {name:>4} " + "  ".join(
                f"{sim_matrix[i,j].item():>6.3f}" for j in range(self.num_concepts)
            )
            print(row)
        
        # Cluster 내부 분산
        print("\n    [Within-Concept Cluster Diversity]")
        for k, name in enumerate(concept_names):
            cluster_p = F.normalize(self.prototypes[k], p=2, dim=1)  # [nc, D]
            intra_sim = torch.mm(cluster_p, cluster_p.t())
            off_diag = intra_sim[~torch.eye(self.num_clusters, dtype=bool, 
                                             device=intra_sim.device)]
            if len(off_diag) > 0:
                print(f"    {name}: avg intra-cluster sim = {off_diag.mean().item():.3f} "
                      f"(lower = more diverse)")
        print()

    # =================================================================
    # Compatibility Methods
    # =================================================================
    def compute_spatial_similarity(self, features):
        """Raw spatial similarity (Scale Factor 미적용)"""
        B, C, H, W = features.shape
        z = F.normalize(features, p=2, dim=1)
        all_p = self.prototypes.view(-1, self.feature_dim)
        all_p = F.normalize(all_p, p=2, dim=1)
        
        all_sim = torch.einsum('bchw,kc->bkhw', z, all_p)
        all_sim = all_sim.view(B, self.num_concepts, self.num_clusters, H, W)
        spatial_sim, _ = all_sim.max(dim=2)  # [B, K, H, W]
        return spatial_sim

    def compute_scores_for_head_training(self, features):
        return self.forward(features.detach())

    def update_with_masks(self, features, masks, update_prototype=True):
        """하위 호환성 유지"""
        concept_scores, _ = self.forward(features)
        return concept_scores

    def save_prototypes(self, path):
        save_dict = {
            'prototypes': self.prototypes.cpu(),
            'logit_scale': self.logit_scale.data.cpu(),
            'initialized': self.initialized.cpu(),
            'num_clusters': self.num_clusters
        }
        torch.save(save_dict, path)
        print(f"[*] Prototypes saved to {path}")
        print(f"    Scale Factor: {self.logit_scale.exp().item():.2f}")
        print(f"    Shape: {list(self.prototypes.shape)} "
              f"({self.num_concepts} concepts × {self.num_clusters} clusters)")

    def load_prototypes(self, path):
        checkpoint = torch.load(path, map_location=self.prototypes.device)
        
        if isinstance(checkpoint, dict) and 'prototypes' in checkpoint:
            loaded = checkpoint['prototypes'].to(self.prototypes.device)
            
            # Shape 호환성 처리
            if loaded.dim() == 2:
                # 이전 버전 [K, D] → [K, 1, D]로 확장
                loaded = loaded.unsqueeze(1).repeat(1, self.num_clusters, 1)
                print("[*] Legacy prototype format detected, expanded to multi-cluster.")
            
            self.prototypes.copy_(loaded)
            
            if 'logit_scale' in checkpoint:
                self.logit_scale.data.copy_(
                    checkpoint['logit_scale'].to(self.prototypes.device)
                )
        else:
            loaded = checkpoint.to(self.prototypes.device)
            if loaded.dim() == 2:
                loaded = loaded.unsqueeze(1).repeat(1, self.num_clusters, 1)
            self.prototypes.copy_(loaded)
            
        self.initialized.fill_(True)
        print(f"[*] Prototypes loaded from {path}")


# --- 테스트 ---
if __name__ == "__main__":
    print("=== PrototypeBank v2 (Multi-Cluster) Test ===\n")
    
    bank = PrototypeBank(num_concepts=4, feature_dim=2048, num_clusters=3)
    
    # 가짜 Prototype 설정
    for k in range(4):
        for c in range(3):
            bank.prototypes[k, c] = F.normalize(torch.randn(2048), p=2, dim=0)
    bank.initialized.fill_(True)
    
    # Forward
    dummy_feat = torch.randn(2, 2048, 7, 7)
    scores, sim_map = bank(dummy_feat)
    
    print(f"Concept Scores: {scores.shape}")       # [2, 8]
    print(f"Spatial Sim Map: {sim_map.shape}")     # [2, 4, 7, 7]
    print(f"Score Dim: {bank.get_score_dim()}")    # 8
    print(f"Prototypes: {bank.prototypes.shape}")  # [4, 3, 2048]
    
    # Orthogonality loss
    orth_loss = bank.get_orthogonality_loss()
    print(f"Orthogonality Loss: {orth_loss.item():.4f}")
    
    bank._print_prototype_similarity()