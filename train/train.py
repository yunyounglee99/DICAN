"""
DICAN Base Training - Enhanced 3-Phase (Separated Loaders)
============================================================
[핵심 변경사항]

★ Phase별 데이터 분리:
  - Phase 1-A: FGADR만 (100% 마스크) → seg 수렴 보장
  - Phase 1-B: DDR+FGADR 전체 → prototype 추출 (데이터 다양성)
  - Phase 1-C: DDR+FGADR 전체 → CBM head 학습 (분류 다양성)

★ Seg Loss 수렴 개선 (기존: 22→20 정체):
  1. Dice smooth: 1.0 → 0.01 (극소 병변 gradient 100배 증가)
  2. Focal BCE: pos_weight=10 추가 (양성 0.01% 불균형 해소)
  3. Dice:Focal 비율: 0.5:0.5 → 0.7:0.3 (양성 학습 강화)
  4. lambda_seg: 50 → 10 (pos_weight가 이미 gradient 증폭)

Phase 1-A: Backbone + TempHead + SegDecoder (FGADR only)
Phase 1-B: Backbone Freeze → Multi-Cluster Prototype 추출 (Full)
Phase 1-C: Enhanced CBM Head 학습 (Full)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class BaseTrainer:
    def __init__(self, args, model, device, loaders):
        """
        Args:
            loaders (dict): {
                'seg_train': FGADR-only train loader (Phase 1-A),
                'seg_val':   FGADR-only val loader   (Phase 1-A),
                'full_train': DDR+FGADR train loader  (Phase 1-B/C),
                'full_val':   DDR+FGADR val loader    (Phase 1-B/C),
            }
        """
        self.args = args
        self.model = model
        self.device = device
        
        # ★ Phase별 다른 loader 사용
        self.seg_train_loader = loaders['seg_train']
        self.seg_val_loader = loaders['seg_val']
        self.full_train_loader = loaders['full_train']
        self.full_val_loader = loaders['full_val']

    def check_data_statistics(self):
        """Phase 1-A (seg) 데이터 통계 출력"""
        print(f"\n[{'='*10} Phase 1-A Data (FGADR Only) {'='*10}]")
        print(f"  Seg Train: {len(self.seg_train_loader.dataset)} samples")
        print(f"  Seg Valid: {len(self.seg_val_loader.dataset)} samples")
        print(f"  Full Train: {len(self.full_train_loader.dataset)} samples")
        print(f"  Full Valid: {len(self.full_val_loader.dataset)} samples")
        
        try:
            batch = next(iter(self.seg_train_loader))
            print(f"\n  [Seg Train Batch Check]")
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
                if 1.0 in unique_vals:
                    print("  ✅ 마스크에 양성 픽셀 존재 (정상)")
                else:
                    print("  ⚠️ 마스크가 전부 0! 경로 확인 필요")
        except Exception as e:
            print(f"  [Warning] {e}")
        print("=" * 45 + "\n")

    # =================================================================
    # Phase 1-A: Backbone + Seg (FGADR Only, 100% Mask Coverage)
    # =================================================================
    def phase_1a_pretrain(self):
        """
        ★ FGADR만 사용 → 배치 100% 마스크 보장 → seg 확실히 수렴
        
        [Seg Loss 수렴 개선]
        1. Dice smooth: 1.0 → 0.01 (극소 병변에서 gradient 100배↑)
        2. Focal BCE + pos_weight=10 (양성 0.01% 불균형 해소)
        3. Dice:Focal = 0.7:0.3 (Dice가 양성 학습의 핵심 드라이버)
        4. lambda_seg: 50 → 10 (pos_weight가 이미 gradient 증폭)
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-A: Backbone + Pixel-Level Seg")
        print(f"  ★ FGADR Only ({len(self.seg_train_loader.dataset)} samples, 100% mask)")
        print(f"  ★ Dice(0.7) + Focal(0.3) | smooth=0.01 | pos_weight=10")
        print(f"  ★ Early Stopping (patience=7)")
        print(f"{'='*60}")
        
        self.model.set_session_mode('pretrain')
        
        trainable_params = self.model.get_trainable_params()
        print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr_base,
            weight_decay=5e-4
        )
        
        epochs = self.args.epochs_base
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        # ★ lambda_seg: 50 → 10 (pos_weight=10이 gradient를 이미 증폭)
        lambda_seg = 10.0
        
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
            total_dice_val = 0.0
            correct = 0
            total = 0
            epoch_masked = 0
            seg_batches = 0
            
            # ★ FGADR-only loader 사용
            loop = tqdm(self.seg_train_loader, desc=f"[1-A] Epoch {epoch+1}/{epochs}")
            
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
                    label_smoothing=0.1
                )
                
                # ─── Seg Loss: Dice(0.7) + Focal(0.3) ───
                # FGADR이므로 대부분 마스크 있음, 그래도 체크
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                
                if has_mask.any():
                    seg_masked = seg_pred[has_mask]
                    masks_masked = masks[has_mask]
                    
                    # ★ 개선된 loss 함수 사용
                    loss_focal = self._focal_bce_loss(seg_masked, masks_masked, gamma=2.0, alpha=0.75)
                    loss_dice = self._dice_loss(seg_masked, masks_masked)
                    
                    # ★ Dice 비중 증가: 양성 학습의 핵심 드라이버
                    loss_seg = 0.3 * loss_focal + 0.7 * loss_dice
                    
                    epoch_masked += has_mask.sum().item()
                    seg_batches += 1
                    total_dice_val += loss_dice.item()
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
                
                # ★ dice loss도 표시
                loop.set_postfix({
                    "cls": f"{loss_cls.item():.3f}",
                    "seg": f"{lambda_seg*loss_seg.item():.3f}",
                    "dice": f"{loss_dice.item():.3f}" if has_mask.any() else "n/a",
                    "acc": f"{100.*correct/total:.1f}%"
                })
            
            scheduler.step()
            train_acc = 100. * correct / total
            avg_seg = total_seg / max(len(self.seg_train_loader), 1)
            avg_dice = total_dice_val / max(seg_batches, 1)
            
            # ★ seg_val도 FGADR val로 평가
            val_acc = self._validate_pretrain(class_weights)
            
            gap = train_acc - val_acc
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"Gap={gap:.1f}%, SegLoss={lambda_seg*avg_seg:.4f}, "
                  f"DiceLoss={avg_dice:.4f}, Masked={epoch_masked}")
            
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
        
        self.load_model("phase1a_best.pth")
        
        # Dice 메트릭 (FGADR val set 대상)
        self._verify_seg_quality_pixel_level()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        return best_val_acc

    def _dice_loss(self, pred, target, smooth=0.01):
        """
        ★ smooth: 1.0 → 0.01
        
        [이유] 양성 픽셀이 85개(MA)일 때:
        - smooth=1.0: dice = (2*0 + 1)/(0+85+1) ≈ 0.012 → loss ≈ 0.99 (상수)
        - smooth=0.01: dice = (2*0 + 0.01)/(0+85+0.01) ≈ 0.0001 → loss ≈ 1.0
        → intersection 변화에 훨씬 민감한 gradient 제공
        """
        pred_sig = torch.sigmoid(pred)
        
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
        """
        ★ pos_weight=10 추가
        
        [이유] 양성 픽셀이 0.01~0.13%로 극도로 희소
        - BCE 자체가 음성에 압도적으로 편향됨
        - pos_weight=10으로 양성 gradient를 10배 증폭
        - 10000:1 불균형 → 1000:1로 개선
        """
        # ★ pos_weight: 양성 픽셀 gradient 10배 증폭
        pos_weight = torch.ones(pred.size(1), device=pred.device) * 10.0
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none',
            pos_weight=pos_weight.view(1, -1, 1, 1)
        )
        
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        focal_weight = (1.0 - pt) ** gamma
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()

    def _validate_pretrain(self, class_weights):
        """★ FGADR val loader로 평가"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in self.seg_val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                outputs = self.model(images)
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100. * correct / total

    def _verify_seg_quality_pixel_level(self):
        """
        ★ FGADR val set 대상 Dice 계산 (100% 마스크 보장)
        """
        print("\n[*] Pixel-Level Segmentation Quality (FGADR Validation Set):")
        self.model.eval()
        self.model.set_session_mode('pretrain')
        self.model.eval()
        
        concept_names = ["EX", "HE", "MA", "SE"]
        dice_scores = {k: [] for k in range(4)}
        iou_scores = {k: [] for k in range(4)}
        
        with torch.no_grad():
            for batch_data in self.seg_val_loader:
                images = batch_data['image'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                if not has_mask.any():
                    continue
                
                feat4, multi_scale = self.model.backbone(images, return_multi_scale=True)
                seg_pred = self.model.seg_decoder(multi_scale)
                seg_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                
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
    # Phase 1-B: Multi-Cluster Prototype Extraction (Full Dataset)
    # =================================================================
    def phase_1b_extract_prototypes(self):
        """
        ★ DDR+FGADR 전체 데이터에서 prototype 추출
        
        [이유]
        - Prototype은 다양한 도메인의 병변 특징을 포괄해야 함
        - DDR의 275장 마스크도 prototype 추출에는 기여 가능
          (backbone이 이미 FGADR로 잘 학습되었으므로)
        - FGADR만으로는 DDR 도메인 특성을 반영 못함
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-B: Multi-Cluster Prototype Extraction")
        print(f"  ★ Full Dataset: DDR+FGADR ({len(self.full_train_loader.dataset)} samples)")
        print(f"  ★ K-Means Clustering (k={self.model.prototypes.num_clusters})")
        print(f"  ★ ROI-Only Feature Extraction")
        print(f"  ★ Gram-Schmidt Orthogonalization")
        print(f"{'='*60}")
        
        self.model.set_session_mode('extract')
        
        # ★ full_train_loader 사용 (DDR+FGADR)
        self.model.prototypes.extract_prototypes_from_dataset(
            backbone=self.model.backbone,
            dataloader=self.full_train_loader,
            device=self.device
        )
        
        save_dir = getattr(self.args, 'save_path', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        proto_path = os.path.join(save_dir, "base_prototypes.pt")
        self.model.prototypes.save_prototypes(proto_path)
        
        print(f"[1-B] Complete.")

    # =================================================================
    # Phase 1-C: Enhanced CBM Head Training (Full Dataset)
    # =================================================================
    def phase_1c_train_head(self):
        """
        ★ DDR+FGADR 전체로 CBM Head 학습
        
        [이유]
        - 분류는 Grade 0~4 전체 분포가 필요
        - DDR이 Grade 분포가 풍부 (6835장, 특히 Grade 0 다수)
        - FGADR만으로는 Grade 0(Normal) 샘플이 부족
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-C: Enhanced CBM Head Training")
        print(f"  ★ Full Dataset: DDR+FGADR ({len(self.full_train_loader.dataset)} samples)")
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
            
            # ★ full_train_loader 사용 (DDR+FGADR)
            loop = tqdm(self.full_train_loader, desc=f"[1-C] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                logits = outputs['logits']
                concept_scores = outputs['concept_scores']
                
                loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
                
                # Sparsity: Grade 0 → 모든 concept 비활성화
                loss_sp = torch.tensor(0.0, device=self.device)
                normal = (labels == 0)
                if normal.any() and concept_scores is not None:
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
            
            # ★ full_val_loader로 평가
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
        """★ full_val_loader로 평가 (DDR+FGADR)"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in self.full_val_loader:
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
            for batch_data in self.full_val_loader:
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
        print(f"  ★ Phase 1-A: FGADR only → Backbone + Seg")
        print(f"  ★ Phase 1-B: DDR+FGADR  → Prototype Extraction")
        print(f"  ★ Phase 1-C: DDR+FGADR  → CBM Head Training")
        print(f"  ★ Seg Fix: smooth=0.01, pos_weight=10, Dice:Focal=0.7:0.3")
        print(f"{'='*60}")
        self.check_data_statistics()
        
        acc_1a = self.phase_1a_pretrain()
        self.phase_1b_extract_prototypes()
        acc_1c = self.phase_1c_train_head()
        
        print(f"\n{'='*60}")
        print(f"  Base Training Complete!")
        print(f"  Phase 1-A (Backbone+PixelSeg, FGADR): {acc_1a:.2f}%")
        print(f"  Phase 1-C (CBM Head, DDR+FGADR):      {acc_1c:.2f}%")
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