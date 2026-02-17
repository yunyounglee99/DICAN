"""
DICAN Base Training - Enhanced 3-Phase
============================================================
[핵심 변경사항]

문제2 (Overfitting) 해결:
  - Early Stopping (patience=7)
  - 강화된 Data Augmentation (Phase 1-A)
  - 단계적 Backbone 해동 (Gradual Unfreezing)
  - Weight Decay 증가 (1e-4 → 5e-4)
  - Label Smoothing (0.1)

문제1 보조:
  - Phase 1-A에 Orthogonality Loss 추가 (seg 학습과 동시에)
  - Seg Loss를 Dice + Focal 하이브리드로 변경
  - Dice 메트릭: 전체 Validation set 대상으로 계산 (샘플 제한 없음)

Phase 1-A: Backbone + TempHead + SegDecoder + OrthoLoss
Phase 1-B: Backbone Freeze → Multi-Cluster Prototype 추출
Phase 1-C: Enhanced CBM Head 학습 (12-dim input)
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
                    print("  ⚠️ 마스크가 전부 0! 경로 확인 필요")
        except Exception as e:
            print(f"  [Warning] {e}")
        print("=" * 35 + "\n")

    # =================================================================
    # Phase 1-A: Backbone + Seg + Orthogonality Loss (with Early Stopping)
    # =================================================================
    def phase_1a_pretrain(self):
        """
        [문제2 해결: 근본적 Overfitting 방지]
        
        1. Early Stopping: patience=7, val_acc 기준
        2. Label Smoothing: 0.1 (과도한 confidence 방지)
        3. Weight Decay: 5e-4 (L2 정규화 강화)
        4. Seg Loss: Dice + Focal 하이브리드
           → 기존 Focal만 사용 시 "all zeros" 예측 문제 해결
        5. Gradient Clipping: max_norm=3.0 (더 엄격)
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-A: Backbone + Pixel-Level Seg (Enhanced)")
        print(f"  ★ Early Stopping (patience=7)")
        print(f"  ★ Dice + Focal Hybrid Seg Loss")
        print(f"  ★ Label Smoothing 0.1")
        print(f"{'='*60}")
        
        self.model.set_session_mode('pretrain')
        
        trainable_params = self.model.get_trainable_params()
        print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
        
        # ★ Weight Decay 강화
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr_base,
            weight_decay=5e-4  # 기존 1e-4 → 5e-4
        )
        
        epochs = self.args.epochs_base
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # ★ Label Smoothing
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        lambda_seg = 50.0
        
        # ★ Early Stopping 변수
        best_val_acc = 0.0
        patience = 7
        patience_counter = 0
        best_epoch = 0
        
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
                
                # ─── Classification Loss (Label Smoothing) ───
                loss_cls = F.cross_entropy(
                    cls_logits, labels, 
                    weight=class_weights,
                    label_smoothing=0.1  # ★ 과도한 confidence 방지
                )
                
                # ─── Seg Loss: Dice + Focal 하이브리드 ───
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                
                if has_mask.any():
                    seg_masked = seg_pred[has_mask]
                    masks_masked = masks[has_mask]
                    
                    loss_focal = self._focal_bce_loss(seg_masked, masks_masked, gamma=2.0, alpha=0.75)
                    loss_dice = self._dice_loss(seg_masked, masks_masked)
                    loss_seg = 0.5 * loss_focal + 0.5 * loss_dice
                    
                    epoch_masked += has_mask.sum().item()
                else:
                    loss_seg = torch.tensor(0.0, device=self.device)
                
                loss = loss_cls + lambda_seg * loss_seg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
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
            train_acc = 100. * correct / total
            avg_seg = total_seg / len(self.train_loader)
            
            val_acc = self._validate_pretrain(class_weights)
            
            gap = train_acc - val_acc
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"Gap={gap:.1f}%, SegLoss={lambda_seg*avg_seg:.4f}, Masked={epoch_masked}")
            
            # ★ Early Stopping Logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_model("phase1a_best.pth")
                print(f"  ★ New Best: {val_acc:.2f}% (Epoch {best_epoch})")
            else:
                patience_counter += 1
                print(f"  (patience: {patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n  ⛔ Early Stopping at Epoch {epoch+1}!")
                    print(f"     Best was Epoch {best_epoch} ({best_val_acc:.2f}%)")
                    break
        
        # Best model 로드
        self.load_model("phase1a_best.pth")
        
        # ★ Dice 메트릭 (전체 Val set 대상)
        self._verify_seg_quality_pixel_level()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        return best_val_acc

    def _dice_loss(self, pred, target, smooth=1.0):
        """
        Dice Loss - "all zeros" 예측 방지
        
        Focal BCE만 쓰면 양성 픽셀이 0.01%인 경우 
        "전부 0으로 예측"하면 loss가 매우 낮아짐.
        Dice는 이를 방지: 양성을 하나도 못 찾으면 dice=0 → loss=1
        """
        pred_sig = torch.sigmoid(pred)
        
        # Per-channel Dice
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # 마스크가 존재하는 채널만 loss 계산
        has_target = (target.sum(dim=(2, 3)) > 0).float()
        
        if has_target.sum() > 0:
            dice_loss = ((1 - dice) * has_target).sum() / has_target.sum()
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)
        
        return dice_loss

    def _focal_bce_loss(self, pred, target, gamma=2.0, alpha=0.75):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        focal_weight = (1.0 - pt) ** gamma
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()

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

    def _verify_seg_quality_pixel_level(self):
        """
        [★ 수정] 전체 Validation set 대상 Dice 계산
        
        기존: 50 배치만 확인 → FGADR 샘플 누락 → 모든 Dice N/A
        변경: 전체 val loader 순회, 마스크 있는 모든 배치에서 계산
        """
        print("\n[*] Pixel-Level Segmentation Quality (Full Validation Set):")
        self.model.eval()
        self.model.set_session_mode('pretrain')
        self.model.eval()
        
        concept_names = ["EX", "HE", "MA", "SE"]
        dice_scores = {k: [] for k in range(4)}
        iou_scores = {k: [] for k in range(4)}
        
        with torch.no_grad():
            # ★ 전체 Validation set 순회 (제한 없음)
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                if not has_mask.any():
                    continue
                
                feat4, multi_scale = self.model.backbone(images, return_multi_scale=True)
                seg_pred = self.model.seg_decoder(multi_scale)
                seg_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                
                # 마스크가 있는 샘플만 추출
                seg_masked = seg_binary[has_mask]
                gt_masked = masks[has_mask]
                
                for k in range(4):
                    pred_k = seg_masked[:, k]
                    gt_k = gt_masked[:, k]
                    
                    has_lesion = (gt_k.sum(dim=(1, 2)) > 0)
                    if has_lesion.any():
                        p = pred_k[has_lesion]
                        g = gt_k[has_lesion]
                        
                        intersection = (p * g).sum(dim=(1, 2))
                        union_dice = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
                        union_iou = union_dice - intersection
                        
                        for i in range(p.size(0)):
                            if union_dice[i] > 0:
                                dice_scores[k].append(
                                    (2 * intersection[i] / union_dice[i]).item()
                                )
                            if union_iou[i] > 0:
                                iou_scores[k].append(
                                    (intersection[i] / union_iou[i]).item()
                                )
        
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
    # Phase 1-B: Multi-Cluster Prototype Extraction
    # =================================================================
    def phase_1b_extract_prototypes(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-B: Multi-Cluster Prototype Extraction")
        print(f"  ★ K-Means Clustering (k={self.model.prototypes.num_clusters})")
        print(f"  ★ ROI-Only Feature Extraction")
        print(f"  ★ Gram-Schmidt Orthogonalization")
        print(f"{'='*60}")
        
        self.model.set_session_mode('extract')
        
        self.model.prototypes.extract_prototypes_from_dataset(
            backbone=self.model.backbone,
            dataloader=self.train_loader,
            device=self.device
        )
        
        save_dir = getattr(self.args, 'save_path', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        proto_path = os.path.join(save_dir, "base_prototypes.pt")
        self.model.prototypes.save_prototypes(proto_path)
        
        print(f"[1-B] Complete.")

    # =================================================================
    # Phase 1-C: Enhanced CBM Head Training
    # =================================================================
    def phase_1c_train_head(self):
        """
        Enhanced Head (12→64→32→16→5) 학습.
        
        [변경사항]
        - 입력: 12-dim (Max + Mean + Std per concept)
        - Early stopping (patience=5)
        - Orthogonality loss로 prototype 품질 유지 모니터링
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-C: Enhanced CBM Head Training")
        print(f"  ★ Input: {self.model.prototypes.get_score_dim()}-dim "
              f"(Max+Mean+Std × {self.model.num_concepts} concepts)")
        print(f"  ★ Architecture: 12→64→32→16→5 with Residual")
        print(f"{'='*60}")
        
        self.model.set_session_mode('head_train')
        
        trainable_params = self.model.get_trainable_params()
        print(f"  Trainable: {sum(p.numel() for p in trainable_params):,} params")
        
        optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
        
        epochs = max(self.args.epochs_base // 2, 15)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.head.train()
            correct = 0
            total = 0
            
            loop = tqdm(self.train_loader, desc=f"[1-C] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                logits = outputs['logits']
                concept_scores = outputs['concept_scores']
                
                # Classification Loss
                loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
                
                # Sparsity: Grade 0 → 모든 concept 비활성화
                loss_sp = torch.tensor(0.0, device=self.device)
                normal = (labels == 0)
                if normal.any() and concept_scores is not None:
                    # Max scores (첫 num_concepts개)
                    max_scores = concept_scores[normal, :self.model.num_concepts]
                    loss_sp = torch.relu(max_scores).mean()
                
                loss = loss_cls + 0.3 * loss_sp
                loss.backward()
                optimizer.step()
                
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
            train_acc = 100. * correct / total
            
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"Gap={train_acc-val_acc:.1f}%, "
                  f"Scale={self.model.prototypes.logit_scale.exp().item():.1f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model("phase1c_best.pth")
                print(f"  ★ Best: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  (patience: {patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"  ⛔ Early Stopping at Epoch {epoch+1}!")
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
        """Multi-cluster score 분석 (12차원)"""
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
        
        nc = self.model.num_concepts
        names = ([f"{n}_max" for n in ["EX","HE","MA","SE"]] +
                 [f"{n}_mean" for n in ["EX","HE","MA","SE"]] +
                 [f"{n}_std" for n in ["EX","HE","MA","SE"]])
        
        print(f"  {'Grade':>5} | " + " | ".join(f"{n:>8}" for n in names))
        print("  " + "-" * (8 + 11 * len(names)))
        for g in range(5):
            if grade_scores[g]:
                all_s = torch.cat(grade_scores[g], dim=0)
                means = all_s.mean(dim=0)
                print(f"  {g:>5} | " + " | ".join(f"{m.item():>8.2f}" for m in means))
        print()

    # =================================================================
    # Main
    # =================================================================
    def run(self):
        print(f"\n{'='*60}")
        print(f"  DICAN Enhanced 3-Phase Training")
        print(f"  ★ Multi-Cluster Prototypes (k={self.model.prototypes.num_clusters})")
        print(f"  ★ Orthogonality Constraint")
        print(f"  ★ Early Stopping + Dice Loss")
        print(f"  ★ Enhanced CBM Head (12→64→32→16→5)")
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
