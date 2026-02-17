"""
DICAN Base Training - 3-Phase (Pixel-Level Seg Supervision)
=============================================================
Phase 1-A: Backbone + TempHead + SegDecoder 학습
Phase 1-B: Backbone Freeze → Masked GAP → Prototype 추출
Phase 1-C: Backbone + Prototype Freeze → CBM Head 학습

★ QWK (Quadratic Weighted Kappa) 추가: 모든 validation에서 Acc와 함께 출력
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    """QWK 계산 유틸리티"""
    if len(y_true) == 0:
        return 0.0
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


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
    # =================================================================
    def phase_1a_pretrain(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-A: Backbone + Pixel-Level Seg (224×224)")
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
        lambda_seg = 100.0
        
        best_val_acc = 0.0
        best_val_kappa = 0.0
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
                
                loss_cls = F.cross_entropy(cls_logits, labels, weight=class_weights)
                
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                
                if has_mask.any():
                    loss_seg = self._focal_bce_loss(
                        seg_pred[has_mask],
                        masks[has_mask],
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
            
            # ★ QWK 포함 Validation
            val_acc, val_kappa = self._validate_pretrain(class_weights)
            
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"QWK={val_kappa:.4f}, "
                  f"SegLoss={lambda_seg*avg_seg:.4f}, MaskedSamples={epoch_masked}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_kappa = val_kappa
                self.save_model("phase1a_best.pth")
                print(f"  ★ Best: {val_acc:.2f}% (QWK={val_kappa:.4f})")
        
        self.load_model("phase1a_best.pth")
        
        print(f"\n  [Info] Total masked samples seen: {total_masked_samples} "
              f"(~{total_masked_samples//max(epochs,1)} per epoch)")
        self._verify_seg_quality_pixel_level()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}%, QWK: {best_val_kappa:.4f}")
        return best_val_acc

    def _focal_bce_loss(self, pred, target, gamma=2.0, alpha=0.75):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        focal_weight = (1.0 - pt) ** gamma
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()

    def _validate_pretrain(self, class_weights):
        """★ Accuracy + QWK 동시 반환"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                outputs = self.model(images)
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = 100. * correct / total
        kappa = quadratic_weighted_kappa(all_labels, all_preds)
        return acc, kappa

    def _verify_seg_quality_pixel_level(self):
        """224×224 해상도에서 Segmentation 품질 검증"""
        print("\n[*] Pixel-Level Segmentation Quality (224×224):")
        self.model.eval()
        self.model.set_session_mode('pretrain')
        self.model.eval()
        
        concept_names = ["EX", "HE", "MA", "SE"]
        dice_scores = {k: [] for k in range(4)}
        iou_scores = {k: [] for k in range(4)}
        
        with torch.no_grad():
            count = 0
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                if not has_mask.any():
                    count += 1
                    if count >= 50:
                        break
                    continue
                
                feat4, multi_scale = self.model.backbone(images, return_multi_scale=True)
                seg_pred = self.model.seg_decoder(multi_scale)
                seg_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                
                for k in range(4):
                    pred_k = seg_binary[:, k]
                    gt_k = masks[:, k]
                    
                    has_lesion = (gt_k.sum(dim=(1, 2)) > 0)
                    if has_lesion.any():
                        p = pred_k[has_lesion]
                        g = gt_k[has_lesion]
                        
                        intersection = (p * g).sum()
                        union_dice = p.sum() + g.sum()
                        union_iou = p.sum() + g.sum() - intersection
                        
                        if union_dice > 0:
                            dice_scores[k].append((2 * intersection / union_dice).item())
                        if union_iou > 0:
                            iou_scores[k].append((intersection / union_iou).item())
                
                count += 1
                if count >= 50:
                    break
        
        print(f"  {'Concept':>8} | {'Dice':>8} | {'IoU':>8} | {'Batches':>8}")
        print(f"  {'-'*40}")
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
    # Phase 1-B: Prototype Extraction
    # =================================================================
    def phase_1b_extract_prototypes(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-B: Prototype Extraction")
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
    # Phase 1-C: Head Training (★ QWK 추가)
    # =================================================================
    def phase_1c_train_head(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-C: CBM Head Training")
        print(f"{'='*60}")
        
        self.model.set_session_mode('head_train')
        
        trainable_params = self.model.get_trainable_params()
        print(f"  Trainable: {sum(p.numel() for p in trainable_params):,} params")
        
        optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
        
        epochs = max(self.args.epochs_base // 2, 15)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        best_val_acc = 0.0
        best_val_kappa = 0.0
        
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
                
                loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
                
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
            
            # ★ QWK 포함 Validation
            val_acc, val_kappa = self._validate_head()
            print(f"  Epoch {epoch+1}: Train={100.*correct/total:.1f}%, Val={val_acc:.1f}%, "
                  f"QWK={val_kappa:.4f}, "
                  f"Scale={self.model.prototypes.logit_scale.exp().item():.1f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_kappa = val_kappa
                self.save_model("phase1c_best.pth")
                print(f"  ★ Best: {val_acc:.2f}% (QWK={val_kappa:.4f})")
        
        self.load_model("phase1c_best.pth")
        self._analyze_concept_scores()
        
        print(f"\n[1-C] Complete. Best Val Acc: {best_val_acc:.2f}%, QWK: {best_val_kappa:.4f}")
        return best_val_acc

    def _validate_head(self):
        """★ Accuracy + QWK 동시 반환"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = 100. * correct / total
        kappa = quadratic_weighted_kappa(all_labels, all_preds)
        return acc, kappa

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
        print()

    # =================================================================
    # Main
    # =================================================================
    def run(self):
        print(f"\n{'='*60}")
        print(f"  DICAN 3-Phase Training (DDR + FGADR)")
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