import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """
    DICAN Prototype Bank - Multi-Cluster + Orthogonality 버전
    
    [핵심 변경사항 - 문제1 (Prototype Collapse) 해결]
    
    1. Multi-Cluster Prototypes: 
       기존: Concept당 1개 prototype → 전부 비슷한 벡터 (0.95+ similarity)
       변경: Concept당 k개 sub-prototype → K-Means 클러스터링으로 
             병변의 다양한 형태를 포착 (예: EX의 soft/hard exudate)
    
    2. Orthogonality Loss:
       loss_orth = || P @ P^T - I ||_F^2
       → 서로 다른 concept은 feature space에서 직교 방향을 가리킴
    
    3. ROI-Only Feature Extraction:
       기존: Masked GAP (mask*feature → 글로벌 feature 오염)
       변경: 마스크 영역의 feature vector만 엄격히 추출
    
    [Score 차원]
    기존: [B, num_concepts * 2] = [B, 8]  (Max + Mean)
    변경: [B, num_concepts * 3] = [B, 12] (Max + Mean + Std)
    """
    
    def __init__(self, num_concepts=4, feature_dim=2048, num_clusters=3, momentum=0.9):
        super(PrototypeBank, self).__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.momentum = momentum

        # Multi-Cluster Prototype: [K, num_clusters, C] = [4, 3, 2048]
        self.register_buffer(
            "prototypes", 
            torch.zeros(num_concepts, num_clusters, feature_dim)
        )
        # Centroid (대표 prototype): [K, C]
        self.register_buffer(
            "proto_centroids",
            torch.zeros(num_concepts, feature_dim)
        )
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        
        # Scale Factor
        self.logit_scale = nn.Parameter(torch.ones([]) * 3.0)
        self.pooling_temp = nn.Parameter(torch.ones([]) * 1.0)
        
        # Score 차원: Max + Mean + Std = 3K
        self.score_dim = num_concepts * 3

    def get_score_dim(self):
        return self.score_dim

    def compute_orthogonality_loss(self):
        """
        [직교성 제약 Loss]
        
        Inter-concept: centroid 간 직교화 (주 목적)
        Intra-concept: 같은 concept 내 sub-proto 간 다양성 장려
        """
        # 1. Inter-concept orthogonality
        p_norm = F.normalize(self.proto_centroids, p=2, dim=1)  # [K, C]
        gram = torch.mm(p_norm, p_norm.t())  # [K, K]
        identity = torch.eye(self.num_concepts, device=gram.device)
        loss_orth = (gram - identity).pow(2).sum()
        
        # 2. Intra-concept diversity
        loss_intra_div = 0.0
        for k in range(self.num_concepts):
            sub_p = F.normalize(self.prototypes[k], p=2, dim=1)  # [nc, C]
            sub_gram = torch.mm(sub_p, sub_p.t())
            sub_identity = torch.eye(self.num_clusters, device=sub_gram.device)
            off_diag = sub_gram * (1 - sub_identity)
            loss_intra_div += off_diag.pow(2).sum()
        
        return loss_orth + 0.3 * loss_intra_div

    def forward(self, features):
        """
        Multi-Cluster Spatial Concept Matching
        
        Args:
            features: [B, 2048, H, W]
        Returns:
            concept_scores: [B, K*3] = [B, 12] (Max + Mean + Std)
            spatial_sim_map: [B, K, H, W]
        """
        batch, ch, h, w = features.shape
        z = F.normalize(features, p=2, dim=1)  # [B, C, H, W]
        scale = self.logit_scale.exp().clamp(max=100.0)
        tau = self.pooling_temp.clamp(min=0.1, max=5.0)
        
        all_lse = []
        all_mean = []
        all_std = []
        spatial_maps = []
        
        for k in range(self.num_concepts):
            sub_protos = F.normalize(self.prototypes[k], p=2, dim=1)  # [nc, C]
            
            # 각 sub-proto와 pixel-level sim: [B, nc, H, W]
            sub_sim = torch.einsum('bchw,nc->bnhw', z, sub_protos)
            
            # Best-matching sub-proto per pixel
            best_sim, _ = sub_sim.max(dim=1)  # [B, H, W]
            scaled_sim = best_sim * scale
            flat_sim = scaled_sim.flatten(1)  # [B, H*W]
            
            score_lse = tau * torch.logsumexp(flat_sim / tau, dim= 1)
            score_mean = flat_sim.mean(dim=1)
            score_std = flat_sim.std(dim=1)
            
            all_lse.append(score_lse)
            all_mean.append(score_mean)
            all_std.append(score_std)
            spatial_maps.append(best_sim.unsqueeze(1))
        
        concept_scores = torch.cat([
            torch.stack(all_lse, dim=1),
            torch.stack(all_mean, dim=1),
            torch.stack(all_std, dim=1),
        ], dim=1)
        
        spatial_sim_map = torch.cat(spatial_maps, dim=1)
        return concept_scores, spatial_sim_map

    # =================================================================
    # Phase 1-B: Multi-Cluster Prototype Extraction
    # =================================================================
    def extract_prototypes_from_dataset(self, backbone, dataloader, device):
        """
        ROI-Only Feature Extraction + K-Means Clustering + Gram-Schmidt 보정
        """
        print("[Phase 1-B] Extracting Multi-Cluster Prototypes...")
        print(f"    Clusters per concept: {self.num_clusters}")
        backbone.eval()
        
        concept_features = {k: [] for k in range(self.num_concepts)}
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                images = batch_data['image'].to(device)
                masks = batch_data['masks'].to(device).float()
                
                features = backbone(images)  # [B, 2048, 7, 7]
                feat_h, feat_w = features.shape[2], features.shape[3]
                
                masks_resized = F.interpolate(
                    masks, size=(feat_h, feat_w), mode='nearest'
                )
                
                for k in range(self.num_concepts):
                    mask_k = masks_resized[:, k, :, :]  # [B, H, W]
                    
                    for b in range(features.size(0)):
                        if mask_k[b].sum() == 0:
                            continue
                        
                        # ★ ROI-Only Extraction
                        feat_b = features[b]       # [2048, 7, 7]
                        mask_b = mask_k[b]         # [7, 7]
                        mask_exp = mask_b.unsqueeze(0)  # [1, 7, 7]
                        
                        masked_feat = feat_b * mask_exp
                        pixel_count = mask_b.sum().clamp(min=1)
                        roi_feature = masked_feat.sum(dim=(1, 2)) / pixel_count
                        roi_feature = F.normalize(roi_feature, p=2, dim=0)
                        concept_features[k].append(roi_feature.cpu())
                
                if (batch_idx + 1) % 50 == 0:
                    counts = {k: len(v) for k, v in concept_features.items()}
                    print(f"    Batch {batch_idx+1}: {counts}")
        
        # K-Means Clustering
        concept_names = ["EX", "HE", "MA", "SE"]
        for k in range(self.num_concepts):
            feats = concept_features[k]
            if len(feats) == 0:
                print(f"    [Warning] Concept {k} ({concept_names[k]}): No samples!")
                continue
            
            feat_tensor = torch.stack(feats, dim=0)
            n_samples = feat_tensor.size(0)
            n_clusters = min(self.num_clusters, n_samples)
            
            centroids = self._kmeans(feat_tensor, n_clusters, max_iter=50)
            
            for c in range(self.num_clusters):
                if c < n_clusters:
                    self.prototypes[k, c] = centroids[c].to(device)
                else:
                    self.prototypes[k, c] = centroids[0].to(device)
            
            centroid = feat_tensor.mean(dim=0)
            self.proto_centroids[k] = F.normalize(centroid, p=2, dim=0).to(device)
            
            print(f"    {concept_names[k]}: {n_samples} samples → {n_clusters} clusters")
        
        self.initialized.fill_(True)
        
        # ★ Gram-Schmidt Soft Orthogonalization
        self._orthogonalize_centroids()
        self._print_prototype_similarity()
        self._print_cluster_diversity()
    
    def _kmeans(self, data, k, max_iter=50):
        n = data.size(0)
        # K-Means++ initialization
        indices = [torch.randint(0, n, (1,)).item()]
        for _ in range(1, k):
            dists = torch.cdist(data, data[indices])
            min_dists = dists.min(dim=1).values
            probs = min_dists / (min_dists.sum() + 1e-8)
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)
        
        centroids = data[indices].clone()
        for _ in range(max_iter):
            dists = torch.cdist(data, centroids)
            assignments = dists.argmin(dim=1)
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = (assignments == c)
                if mask.any():
                    new_centroids[c] = data[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return F.normalize(centroids, p=2, dim=1)
    
    def _orthogonalize_centroids(self):
        """Gram-Schmidt soft orthogonalization (50% blend)"""
        print("\n    [*] Applying soft orthogonalization...")
        
        centroids = self.proto_centroids.clone()
        ortho = torch.zeros_like(centroids)
        
        for i in range(self.num_concepts):
            v = centroids[i].clone()
            for j in range(i):
                proj = torch.dot(v, ortho[j]) * ortho[j]
                v = v - proj
            ortho[i] = F.normalize(v, p=2, dim=0)
        
        alpha = 0.5
        blended = alpha * F.normalize(centroids, p=2, dim=1) + (1 - alpha) * ortho
        self.proto_centroids.copy_(F.normalize(blended, p=2, dim=1))
        
        for k in range(self.num_concepts):
            delta = self.proto_centroids[k] - F.normalize(centroids[k], p=2, dim=0)
            for c in range(self.num_clusters):
                self.prototypes[k, c] = F.normalize(
                    self.prototypes[k, c] + 0.3 * delta, p=2, dim=0
                )

    def _print_prototype_similarity(self):
        p = F.normalize(self.proto_centroids, p=2, dim=1)
        sim_matrix = torch.mm(p, p.t())
        concept_names = ["EX", "HE", "MA", "SE"]
        
        print("\n    [Prototype Centroid Similarity Matrix]")
        header = "         " + "  ".join(f"{n:>6}" for n in concept_names)
        print(header)
        for i, name in enumerate(concept_names):
            row = f"    {name:>4} " + "  ".join(
                f"{sim_matrix[i,j].item():>6.3f}" for j in range(self.num_concepts)
            )
            print(row)
        
        mask = 1 - torch.eye(self.num_concepts, device=sim_matrix.device)
        avg_off = (sim_matrix * mask).sum() / mask.sum()
        print(f"\n    Avg off-diagonal similarity: {avg_off.item():.4f}")
        print(f"    (Target: < 0.3, Previous: ~0.95)")
    
    def _print_cluster_diversity(self):
        concept_names = ["EX", "HE", "MA", "SE"]
        print("\n    [Intra-Concept Cluster Diversity]")
        for k in range(self.num_concepts):
            sub_p = F.normalize(self.prototypes[k], p=2, dim=1)
            sim = torch.mm(sub_p, sub_p.t())
            n = self.num_clusters
            off_mask = 1 - torch.eye(n, device=sim.device)
            avg_sim = (sim * off_mask).sum() / off_mask.sum()
            print(f"    {concept_names[k]}: Avg sub-proto similarity = {avg_sim.item():.4f}")

    def compute_scores_for_head_training(self, features):
        return self.forward(features.detach())

    def update_with_masks(self, features, masks, update_prototype=True):
        concept_scores, _ = self.forward(features)
        return concept_scores

    def save_prototypes(self, path):
        save_dict = {
            'prototypes': self.prototypes.cpu(),
            'proto_centroids': self.proto_centroids.cpu(),
            'logit_scale': self.logit_scale.data.cpu(),
            'initialized': self.initialized.cpu(),
            'num_clusters': self.num_clusters
        }
        torch.save(save_dict, path)
        print(f"[*] Multi-Cluster Prototypes saved to {path}")
        print(f"    Scale Factor: {self.logit_scale.exp().item():.2f}")

    def load_prototypes(self, path):
        checkpoint = torch.load(path, map_location=self.prototypes.device)
        if isinstance(checkpoint, dict) and 'prototypes' in checkpoint:
            self.prototypes.copy_(checkpoint['prototypes'].to(self.prototypes.device))
            if 'proto_centroids' in checkpoint:
                self.proto_centroids.copy_(
                    checkpoint['proto_centroids'].to(self.prototypes.device)
                )
            if 'logit_scale' in checkpoint:
                self.logit_scale.data.copy_(
                    checkpoint['logit_scale'].to(self.prototypes.device)
                )
        else:
            old_proto = checkpoint.to(self.prototypes.device)
            for k in range(self.num_concepts):
                for c in range(self.num_clusters):
                    self.prototypes[k, c] = old_proto[k]
                self.proto_centroids[k] = old_proto[k]
        self.initialized.fill_(True)
        print(f"[*] Prototypes loaded from {path}")
