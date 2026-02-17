"""
DICAN Base Training v2 - 핵심 문제 3건 해결
================================================================

[Problem 1 Fix] Prototype 유사도 0.95→0.5 이하
  - Phase 1-A: Dice+Focal loss → backbone이 병변 type별 feature 차별화 학습
  - Phase 1-B: Multi-cluster extraction + 직교화 (prototypes.py에서 처리)
  - Phase 1-C: Orthogonality regularization 추가

[Problem 2 Fix] Dice/IoU 전부 N/A
  - Seg loss: Focal BCE → Dice + Focal BCE combined (class imbalance 해결)
  - Dice eval: 50 batch 제한 제거 → 전체 validation set 순회
  - ConcatDataset에서 FGADR이 뒤에 오는 문제 해결

[Problem 4 Fix] Phase 1-A 77.6% → Phase 1-C 69.5% (성능 하락)
  - Enhanced head (head.py에서 처리)
  - Phase 1-C에서 warmup + cosine scheduling
  - Phase 1-A에서 early stopping (과적합 방지)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class BaseTrainer:
    def __init__(self, args, model, device, train_loader, val_loader):
        self.args = args
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def check_data_statistics(self):
        """데이터 통계 + 마스크 커버리지 출력"""
        print(f"\n[{'='*10} Data Statistics {'='*10}]")
        print(f"  Train: {len(self.train_loader.dataset)} samples")
        print(f"  Valid: {len(self.val_loader.dataset)} samples")
        
        try:
            batch = next(iter(self.train_loader))
            print(f"  Image: {batch['image'].shape}")
            if 'masks' in batch:
                masks = batch['masks']
                print(f"  Masks: {masks.shape}")
                
                has_any_mask = (masks.sum(dim=(1,2,3)) > 0).sum().item()
                print(f"  Batch mask coverage: {has_any_mask}/{masks.size(0)} images "
                      f"({100*has_any_mask/masks.size(0):.0f}%)")
                
                for k, name in enumerate(["EX", "HE", "MA", "SE"]):
                    active_imgs = (masks[:, k].sum(dim=(1,2)) > 0).sum().item()
                    active_pixels = (masks[:, k] > 0).sum().item()
                    total_pixels = masks[:, k].numel()
                    ratio = 100.0 * active_pixels / total_pixels
                    print(f"    {name}: {active_imgs}/{masks.size(0)} images, "
                          f"{active_pixels:,} pixels ({ratio:.2f}%)")
                
                unique_vals = torch.unique(masks)
                print(f"  Mask unique values: {unique_vals.tolist()}")
                if 1.0 in unique_vals:
                    print("  ✅ 마스크에 양성 픽셀 존재 (정상)")
                else:
                    print("  ⚠️ 마스크가 전부 0! 경로/로딩 문제 확인 필요")
                    
        except Exception as e:
            print(f"  [Warning] {e}")
        print("=" * 35 + "\n")

    # =================================================================
    # Phase 1-A: Backbone + Pixel-Level Segmentation
    # ★ Dice + Focal BCE Combined Loss
    # ★ Early Stopping (patience=7)
    # =================================================================
    def phase_1a_pretrain(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-A: Backbone + Pixel-Level Seg (Dice+Focal)")
        print(f"{'='*60}")
        
        self.model.set_session_mode('pretrain')
        
        trainable_params = self.model.get_trainable_params()
        print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr_base,
            weight_decay=self.args.weight_decay
        )
        
        epochs = self.args.epochs_base
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        # ★ Seg Loss 설정
        lambda_seg = 50.0     # Dice loss는 이미 [0,1] 범위이므로 조정
        dice_weight = 1.0     # Dice loss 비율
        focal_weight = 0.5    # Focal BCE 비율
        
        best_val_acc = 0.0
        patience = 7
        patience_counter = 0
        total_masked_samples = 0
        
        for epoch in range(epochs):
            self.model.train()
            self.model.projector.eval()
            self.model.head.eval()
            
            total_loss = 0.0
            total_cls = 0.0
            total_seg = 0.0
            correct = 0
            total = 0
            epoch_masked = 0
            
            loop = tqdm(self.train_loader, desc=f"[1-A] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                cls_logits = outputs['logits']
                seg_pred = outputs['seg_pred']
                
                # ─── Classification Loss ───
                loss_cls = F.cross_entropy(cls_logits, labels, weight=class_weights)
                
                # ─── Seg Loss: Dice + Focal BCE ───
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                
                if has_mask.any():
                    loss_seg = self._dice_focal_loss(
                        seg_pred[has_mask],
                        masks[has_mask],
                        dice_weight=dice_weight,
                        focal_weight=focal_weight,
                        gamma=2.0, alpha=0.75
                    )
                    epoch_masked += has_mask.sum().item()
                else:
                    loss_seg = torch.tensor(0.0, device=self.device)
                
                loss = loss_cls + lambda_seg * loss_seg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_cls += loss_cls.item()
                total_seg += loss_seg.item()
                
                _, predicted = cls_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                loop.set_postfix({
                    "cls": f"{loss_cls.item():.3f}",
                    "seg": f"{lambda_seg*loss_seg.item():.3f}",
                    "acc": f"{100.*correct/total:.1f}%"
                })
            
            scheduler.step()
            total_masked_samples += epoch_masked
            train_acc = 100. * correct / total
            avg_seg = total_seg / len(self.train_loader)
            
            val_acc = self._validate_pretrain(class_weights)
            
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"SegLoss={lambda_seg*avg_seg:.4f}, MaskedSamples={epoch_masked}")
            
            # ★ Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model("phase1a_best.pth")
                print(f"  ★ Best: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  ⏹ Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} epochs)")
                    break
        
        self.load_model("phase1a_best.pth")
        
        print(f"\n  [Info] Total masked samples seen: {total_masked_samples} "
              f"(~{total_masked_samples//max(epoch+1,1)} per epoch)")
        self._verify_seg_quality_pixel_level()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}%")
        return best_val_acc

    # =================================================================
    # ★★★ Dice + Focal BCE Combined Loss ★★★
    # =================================================================
    def _dice_focal_loss(self, pred, target, dice_weight=1.0, focal_weight=0.5,
                         gamma=2.0, alpha=0.75):
        """
        [핵심 수정: Focal BCE만으로는 극단적 class imbalance 해결 불가]
        
        양성 픽셀 비율: 0.01~0.04%
        → Focal BCE: "모두 0 예측" 시에도 loss가 매우 낮음
        → Dice Loss: 양성/음성 비율에 무관하게 overlap 자체를 최적화
        
        Combined Loss = dice_weight * Dice + focal_weight * Focal_BCE
        
        Dice Loss 특성:
        - 범위: [0, 1], 값이 작을수록 좋음
        - "모두 0 예측" 시 dice = 0 → loss = 1.0 (높은 페널티!)
        - 양성 영역과의 overlap을 직접 최대화
        
        Per-channel Dice:
        - 각 concept(EX/HE/MA/SE)별로 독립적으로 Dice 계산
        - 빈 mask인 channel은 제외 (loss에 노이즈 추가 방지)
        """
        # ─── Dice Loss (per-channel, per-sample) ───
        pred_sigmoid = torch.sigmoid(pred)  # [B, 4, 224, 224]
        smooth = 1.0
        
        # Channel별 Dice 계산
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))  # [B, 4]
        cardinality = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # [B, 4]
        
        dice_per_channel = (2.0 * intersection + smooth) / (cardinality + smooth)  # [B, 4]
        
        # ★ 실제 양성 픽셀이 있는 channel만 Dice loss에 포함
        has_positive = (target.sum(dim=(2, 3)) > 0).float()  # [B, 4]
        
        if has_positive.sum() > 0:
            # 양성 있는 channel의 dice만 평균
            dice_loss = 1.0 - (dice_per_channel * has_positive).sum() / has_positive.sum().clamp(min=1)
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)
        
        # ─── Focal BCE Loss ───
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = pred_sigmoid * target + (1.0 - pred_sigmoid) * (1.0 - target)
        focal_weight_map = (1.0 - pt) ** gamma
        alpha_weight = alpha * target + (1.0 - alpha) * (1.0 - target)
        focal_loss = (alpha_weight * focal_weight_map * bce).mean()
        
        # ─── Combined ───
        return dice_weight * dice_loss + focal_weight * focal_loss

    def _validate_pretrain(self, class_weights):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                outputs = self.model(images)
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    # =================================================================
    # ★★★ Fixed Dice/IoU Evaluation (전체 validation 순회) ★★★
    # =================================================================
    def _verify_seg_quality_pixel_level(self):
        """
        [v1 문제점]
        - 50 batch 후 중단 → ConcatDataset에서 DDR(2503개)이 먼저 나와서
          FGADR validation(276개, 100% 마스크)에 도달하기 전에 종료
        - 결과: Batches=0, Dice=N/A
        
        [v2 수정]
        - 전체 validation set 순회 (제한 없음)
        - 마스크가 있는 배치만 누적하여 통계 계산
        """
        print("\n[*] Pixel-Level Segmentation Quality (224×224):")
        self.model.eval()
        self.model.set_session_mode('pretrain')
        self.model.eval()
        
        concept_names = ["EX", "HE", "MA", "SE"]
        dice_scores = {k: [] for k in range(4)}
        iou_scores = {k: [] for k in range(4)}
        total_batches_with_mask = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:  # ★ 전체 순회
                images = batch_data['image'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                if not has_mask.any():
                    continue
                
                total_batches_with_mask += 1
                
                feat4, multi_scale = self.model.backbone(images, return_multi_scale=True)
                seg_pred = self.model.seg_decoder(multi_scale)
                seg_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                
                for k in range(4):
                    pred_k = seg_binary[:, k]   # [B, 224, 224]
                    gt_k = masks[:, k]           # [B, 224, 224]
                    
                    has_lesion = (gt_k.sum(dim=(1, 2)) > 0)
                    if has_lesion.any():
                        p = pred_k[has_lesion]
                        g = gt_k[has_lesion]
                        
                        intersection = (p * g).sum(dim=(1, 2))  # [N]
                        union_dice = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
                        union_iou = union_dice - intersection
                        
                        # Per-sample Dice & IoU
                        for i in range(intersection.shape[0]):
                            if union_dice[i] > 0:
                                dice_scores[k].append(
                                    (2 * intersection[i] / union_dice[i]).item()
                                )
                            if union_iou[i] > 0:
                                iou_scores[k].append(
                                    (intersection[i] / union_iou[i]).item()
                                )
        
        print(f"  Total batches with masks: {total_batches_with_mask}")
        print(f"  {'Concept':>8} | {'Dice':>8} | {'IoU':>8} | {'Samples':>8}")
        print(f"  {'-'*42}")
        for k, name in enumerate(concept_names):
            if dice_scores[k]:
                d = np.mean(dice_scores[k])
                i = np.mean(iou_scores[k])
                n = len(dice_scores[k])
                print(f"  {name:>8} | {d:>8.4f} | {i:>8.4f} | {n:>8}")
            else:
                print(f"  {name:>8} | {'N/A':>8} | {'N/A':>8} | {'0':>8}")
        print()

    # =================================================================
    # Phase 1-B: Prototype Extraction (Multi-Cluster)
    # =================================================================
    def phase_1b_extract_prototypes(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-B: Multi-Cluster Prototype Extraction")
        print(f"{'='*60}")
        
        self.model.set_session_mode('extract')
        
        # ★ prototypes.py 내부에서 k-means + 직교화 수행
        self.model.prototypes.extract_prototypes_from_dataset(
            backbone=self.model.backbone,
            dataloader=self.train_loader,
            device=self.device
        )
        
        save_dir = getattr(self.args, 'save_path', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        proto_path = os.path.join(save_dir, "base_prototypes.pt")
        self.model.prototypes.save_prototypes(proto_path)
        
        # 직교성 확인
        orth_loss = self.model.prototypes.get_orthogonality_loss()
        print(f"    Orthogonality Loss (lower=better): {orth_loss.item():.4f}")
        print(f"    (Perfect orthogonality = 0.0)")
        
        print(f"[1-B] Complete.")

    # =================================================================
    # Phase 1-C: Enhanced CBM Head Training
    # ★ Orthogonality regularization
    # ★ Warmup + Cosine scheduling
    # =================================================================
    def phase_1c_train_head(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-C: Enhanced CBM Head Training")
        print(f"{'='*60}")
        
        self.model.set_session_mode('head_train')
        
        trainable_params = self.model.get_trainable_params()
        n_params = sum(p.numel() for p in trainable_params)
        print(f"  Trainable: {n_params:,} params")
        print(f"  Head architecture: Feature Interaction + ResidualBlock")
        
        # ★ AdamW + warmup cosine scheduler
        epochs = max(self.args.epochs_base // 2, 15)
        warmup_epochs = min(3, epochs // 5)
        
        optimizer = optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-4)
        
        # Warmup + Cosine
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        # ★ Orthogonality regularization 가중치
        lambda_orth = 0.1
        
        best_val_acc = 0.0
        patience = 7
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.head.train()
            correct = 0
            total = 0
            epoch_orth_loss = 0.0
            
            loop = tqdm(self.train_loader, desc=f"[1-C] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                logits = outputs['logits']
                concept_scores = outputs['concept_scores']
                
                # ─── Classification Loss ───
                loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
                
                # ─── Sparsity: Grade 0은 concept 비활성화 ───
                loss_sp = torch.tensor(0.0, device=self.device)
                normal = (labels == 0)
                if normal.any() and concept_scores is not None:
                    max_scores = concept_scores[normal, :self.model.num_concepts]
                    loss_sp = torch.relu(max_scores).mean()
                
                # ★ Orthogonality Regularization
                loss_orth = self.model.prototypes.get_orthogonality_loss()
                
                loss = loss_cls + 0.3 * loss_sp + lambda_orth * loss_orth
                loss.backward()
                optimizer.step()
                
                epoch_orth_loss += loss_orth.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                loop.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "acc": f"{100.*correct/total:.1f}%",
                    "scale": f"{self.model.prototypes.logit_scale.exp().item():.1f}"
                })
            
            scheduler.step()
            val_acc = self._validate_head()
            avg_orth = epoch_orth_loss / len(self.train_loader)
            
            print(f"  Epoch {epoch+1}: Train={100.*correct/total:.1f}%, Val={val_acc:.1f}%, "
                  f"Scale={self.model.prototypes.logit_scale.exp().item():.1f}, "
                  f"Orth={avg_orth:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model("phase1c_best.pth")
                print(f"  ★ Best: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  ⏹ Early stopping at epoch {epoch+1}")
                    break
        
        self.load_model("phase1c_best.pth")
        self._analyze_concept_scores()
        
        print(f"\n[1-C] Complete. Best Val Acc: {best_val_acc:.2f}%")
        return best_val_acc

    def _validate_head(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                features = self.model.backbone(images)
                scores, _ = self.model.prototypes(features)
                logits = self.model.head(scores)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    def _analyze_concept_scores(self):
        print("\n[*] Concept Score Analysis by DR Grade:")
        self.model.eval()
        
        grade_scores = {g: [] for g in range(5)}
        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                features = self.model.backbone(images)
                scores, _ = self.model.prototypes(features)
                for g in range(5):
                    m = (labels == g)
                    if m.any():
                        grade_scores[g].append(scores[m].cpu())
        
        names = ["EX_max","HE_max","MA_max","SE_max",
                 "EX_mean","HE_mean","MA_mean","SE_mean"]
        
        print(f"  {'Grade':>5} | " + " | ".join(f"{n:>7}" for n in names))
        print("  " + "-" * (8 + 10 * len(names)))
        for g in range(5):
            if grade_scores[g]:
                all_s = torch.cat(grade_scores[g], dim=0)
                means = all_s.mean(dim=0)
                print(f"  {g:>5} | " + " | ".join(f"{m.item():>7.2f}" for m in means))
        
        # ★ Concept Score 변별력 분석
        print("\n  [Concept Discriminability]")
        if grade_scores[0] and grade_scores[3]:
            g0 = torch.cat(grade_scores[0], dim=0).mean(0)
            g3 = torch.cat(grade_scores[3], dim=0).mean(0)
            diff = g3 - g0
            for i, name in enumerate(names):
                print(f"    {name}: Grade3-Grade0 = {diff[i].item():+.2f}")
        print()

    # =================================================================
    # Main Pipeline
    # =================================================================
    def run(self):
        print(f"\n{'='*60}")
        print(f"  DICAN v2 3-Phase Training")
        print(f"  ★ Dice+Focal Loss | Multi-Cluster Proto | Enhanced Head")
        print(f"{'='*60}")
        self.check_data_statistics()
        
        acc_1a = self.phase_1a_pretrain()
        self.phase_1b_extract_prototypes()
        acc_1c = self.phase_1c_train_head()
        
        print(f"\n{'='*60}")
        print(f"  Base Training Complete!")
        print(f"  Phase 1-A (Backbone+PixelSeg): {acc_1a:.2f}%")
        print(f"  Phase 1-C (CBM Head):          {acc_1c:.2f}%")
        print(f"{'='*60}\n")
        
        return self.model

    def save_model(self, filename):
        d = getattr(self.args, 'save_path', './checkpoints')
        os.makedirs(d, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(d, filename))

    def load_model(self, filename):
        d = getattr(self.args, 'save_path', './checkpoints')
        path = os.path.join(d, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)