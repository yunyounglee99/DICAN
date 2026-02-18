"""
DICAN Base Training - DDR+FGADR 통합 세팅 (원래 세팅 복원)
============================================================
★ Phase 1-A: DDR+FGADR 전체 → Backbone + Seg + Classification
  - DDR 6260장: cls 100% 기여 + seg 4.4% 기여
  - FGADR 1566장: cls + seg 100% 기여
  - Backbone이 다양한 도메인의 DR 특징을 학습 → 높은 feature 품질

★ Phase 1-B: DDR+FGADR 전체 → Prototype Extraction
★ Phase 1-C: DDR+FGADR 전체 → CBM Head Training
★ QWK 평가, DRAC22 Soft Label, adaptation_steps=20
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score


class BaseTrainer:
    def __init__(self, args, model, device, loaders):
        """
        Args:
            loaders (dict): {
                'seg_train': FGADR-only train loader (Dice 평가용),
                'seg_val':   FGADR-only val loader   (Dice 평가용),
                'full_train': DDR+FGADR train loader  (★ Phase 1-A/B/C 모두 사용),
                'full_val':   DDR+FGADR val loader    (★ Phase 1-A/B/C 모두 사용),
            }
        """
        self.args = args
        self.model = model
        self.device = device
        
        # FGADR-only (Dice 평가 전용)
        self.seg_train_loader = loaders['seg_train']
        self.seg_val_loader = loaders['seg_val']
        
        # ★ DDR+FGADR 통합 (Phase 1-A/B/C 모두 사용)
        self.full_train_loader = loaders['full_train']
        self.full_val_loader = loaders['full_val']

    def check_data_statistics(self):
        """★ Phase 1-A 데이터 통계 (DDR+FGADR 통합)"""
        print(f"\n[{'='*10} Data Statistics {'='*10}]")
        print(f"  Train: {len(self.full_train_loader.dataset)} samples")
        print(f"  Valid: {len(self.full_val_loader.dataset)} samples")
        
        try:
            batch = next(iter(self.full_train_loader))
            print(f"\n  [Train Batch Check]")
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
                
                if 1.0 in torch.unique(masks):
                    print("  ✅ 마스크에 양성 픽셀 존재 (정상)")
                else:
                    print("  ⚠️ 마스크가 전부 0! 경로 확인 필요")
        except Exception as e:
            print(f"  [Warning] {e}")
        print("=" * 45 + "\n")

    # =================================================================
    # Phase 1-A: Backbone + Seg (★ DDR+FGADR 통합)
    # =================================================================
    def phase_1a_pretrain(self):
        """
        ★ DDR+FGADR 통합 사용 (원래 세팅)
        
        - DDR 6260장: classification에 100% 기여, seg에 4.4% 기여
        - FGADR 1566장: 둘 다 100% 기여
        - Backbone이 다양한 도메인 + Grade 분포를 학습하여 
          feature 품질이 높아짐 → 이후 Phase 1-C에서 71% 달성
        """
        n_train = len(self.full_train_loader.dataset)
        
        print(f"\n{'='*60}")
        print(f"  Phase 1-A: Backbone + Pixel-Level Seg (Enhanced)")
        print(f"  ★ DDR+FGADR ({n_train} samples)")
        print(f"  ★ Dice + Focal Hybrid Seg Loss")
        print(f"  ★ Early Stopping + Dice Loss")
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
        lambda_seg = 10.0
        
        best_val_acc = 0.0
        patience = 7
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            self.model.train()
            self.model.projector.eval()
            self.model.head.eval()
            
            total_dice_val = 0.0
            correct = 0
            total = 0
            epoch_masked = 0
            seg_batches = 0
            total_seg = 0.0
            
            # ★ DDR+FGADR 통합 loader 사용
            loop = tqdm(self.full_train_loader, desc=f"[1-A] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                cls_logits = outputs['logits']
                seg_pred = outputs['seg_pred']
                
                # ─── Classification Loss (모든 샘플) ───
                loss_cls = F.cross_entropy(
                    cls_logits, labels, 
                    weight=class_weights,
                    label_smoothing=0.1
                )
                
                # ─── Seg Loss (마스크 있는 샘플만) ───
                has_mask = (masks.sum(dim=(1, 2, 3)) > 0)
                
                if has_mask.any():
                    seg_masked = seg_pred[has_mask]
                    masks_masked = masks[has_mask]
                    
                    loss_focal = self._focal_bce_loss(seg_masked, masks_masked)
                    loss_dice = self._dice_loss(seg_masked, masks_masked)
                    loss_seg = 0.3 * loss_focal + 0.7 * loss_dice
                    
                    epoch_masked += has_mask.sum().item()
                    seg_batches += 1
                    total_dice_val += loss_dice.item()
                else:
                    loss_seg = torch.tensor(0.0, device=self.device)
                    loss_dice = loss_seg
                
                loss = loss_cls + lambda_seg * loss_seg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
                optimizer.step()
                
                total_seg += loss_seg.item()
                
                _, predicted = cls_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                loop.set_postfix({
                    "cls": f"{loss_cls.item():.3f}",
                    "seg": f"{lambda_seg*loss_seg.item():.3f}",
                    "dice": f"{loss_dice.item():.3f}" if has_mask.any() else "n/a",
                    "acc": f"{100.*correct/total:.1f}%"
                })
            
            scheduler.step()
            train_acc = 100. * correct / total
            avg_seg = total_seg / max(len(self.full_train_loader), 1)
            avg_dice = total_dice_val / max(seg_batches, 1)
            
            # ★ DDR+FGADR 통합 val loader로 평가
            val_acc, val_qwk = self._validate_pretrain()
            
            gap = train_acc - val_acc
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"QWK={val_qwk:.4f}, Gap={gap:.1f}%, "
                  f"SegLoss={lambda_seg*avg_seg:.4f}, Masked={epoch_masked}")
            
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
        
        # Dice 평가는 FGADR val set으로 (100% 마스크)
        self._verify_seg_quality_pixel_level()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        return best_val_acc

    def _dice_loss(self, pred, target, smooth=0.01):
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        has_target = (target.sum(dim=(2, 3)) > 0).float()
        if has_target.sum() > 0:
            return ((1 - dice) * has_target).sum() / has_target.sum()
        return torch.tensor(0.0, device=pred.device)

    def _focal_bce_loss(self, pred, target, gamma=2.0, alpha=0.75):
        pos_weight = torch.ones(pred.size(1), device=pred.device) * 10.0
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none',
            pos_weight=pos_weight.view(1, -1, 1, 1)
        )
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        focal_weight = (1.0 - pt) ** gamma
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        return (alpha_weight * focal_weight * bce).mean()

    def _validate_pretrain(self):
        """★ DDR+FGADR 통합 val loader로 평가"""
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_data in self.full_val_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                outputs = self.model(images)
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = 100. * correct / total
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic') if total > 0 else 0.0
        return acc, qwk

    def _verify_seg_quality_pixel_level(self):
        """FGADR val set 대상 Dice 계산 (100% 마스크 보장)"""
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
                                dice_scores[k].append((2 * intersection[i] / union_dice[i]).item())
                            if union_iou[i] > 0:
                                iou_scores[k].append((intersection[i] / union_iou[i]).item())
        
        print(f"  {'Concept':>8} | {'Dice':>8} | {'IoU':>8} | {'Samples':>8}")
        print(f"  {'-'*42}")
        for k, name in enumerate(concept_names):
            if dice_scores[k]:
                print(f"  {name:>8} | {np.mean(dice_scores[k]):>8.4f} | "
                      f"{np.mean(iou_scores[k]):>8.4f} | {len(dice_scores[k]):>8}")
            else:
                print(f"  {name:>8} | {'N/A':>8} | {'N/A':>8} | {'0':>8}")
        print()

    # =================================================================
    # Phase 1-B: Multi-Cluster Prototype Extraction (Full Dataset)
    # =================================================================
    def phase_1b_extract_prototypes(self):
        print(f"\n{'='*60}")
        print(f"  Phase 1-B: Multi-Cluster Prototype Extraction")
        print(f"  ★ Full Dataset: DDR+FGADR ({len(self.full_train_loader.dataset)} samples)")
        print(f"  ★ K-Means Clustering (k={self.model.prototypes.num_clusters})")
        print(f"  ★ ROI-Only Feature Extraction")
        print(f"  ★ Gram-Schmidt Orthogonalization")
        print(f"{'='*60}")
        
        self.model.set_session_mode('extract')
        
        self.model.prototypes.extract_prototypes_from_dataset(
            backbone=self.model.backbone,
            dataloader=self.full_train_loader,
            device=self.device
        )
        
        save_dir = getattr(self.args, 'save_path', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        self.model.prototypes.save_prototypes(os.path.join(save_dir, "base_prototypes.pt"))
        
        print(f"[1-B] Complete.")

    # =================================================================
    # Phase 1-C: Enhanced CBM Head Training (Full Dataset)
    # =================================================================
    def phase_1c_train_head(self):
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
        best_val_qwk = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.head.train()
            correct, total = 0, 0
            
            loop = tqdm(self.full_train_loader, desc=f"[1-C] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                logits = outputs['logits']
                concept_scores = outputs['concept_scores']
                
                loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
                
                # Sparsity: Grade 0 → concept 비활성화
                loss_sp = torch.tensor(0.0, device=self.device)
                normal = (labels == 0)
                if normal.any() and concept_scores is not None:
                    loss_sp = torch.relu(concept_scores[normal, :self.model.num_concepts]).mean()
                
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
            
            val_acc, val_qwk = self._validate_head()
            train_acc = 100. * correct / total
            
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"QWK={val_qwk:.4f}, Gap={train_acc-val_acc:.1f}%, "
                  f"Scale={self.model.prototypes.logit_scale.exp().item():.1f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_qwk = val_qwk
                patience_counter = 0
                self.save_model("phase1c_best.pth")
                print(f"  ★ Best: {val_acc:.2f}% (QWK={val_qwk:.4f})")
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
        correct, total = 0, 0
        all_preds, all_labels = [], []
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
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = 100. * correct / total
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic') if total > 0 else 0.0
        return acc, qwk

    def _analyze_concept_scores(self):
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
        print(f"  DICAN 3-Phase Training (DDR+FGADR 통합)")
        print(f"  ★ Phase 1-A: DDR+FGADR → Backbone + Seg")
        print(f"  ★ Phase 1-B: DDR+FGADR → Prototype Extraction")
        print(f"  ★ Phase 1-C: DDR+FGADR → CBM Head Training")
        print(f"  ★ Seg Fix: smooth=0.01, pos_weight=10, Dice:Focal=0.7:0.3")
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
        self.model.load_state_dict(
            torch.load(os.path.join(d, filename), map_location=self.device), strict=False)


# =================================================================
# Incremental Loader Manager
# =================================================================
class IncrementalLoaderManager:
    """Incremental Session 데이터 로더 관리"""
    def __init__(self, data_root, n_shot=10, batch_size=32, seed=42):
        self.data_root = data_root
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.seed = seed

    def get_incremental_loaders(self, task_id, mode_override=None):
        from data.inc_loader import get_incremental_loader
        
        if mode_override == 'test':
            test_loader = get_incremental_loader(
                session_id=task_id, data_root=self.data_root,
                mode='test', batch_size=self.batch_size, shot=None)
            return None, test_loader
        
        support_loader = get_incremental_loader(
            session_id=task_id, data_root=self.data_root,
            mode='train', batch_size=self.batch_size, shot=self.n_shot)
        query_loader = get_incremental_loader(
            session_id=task_id, data_root=self.data_root,
            mode='test', batch_size=self.batch_size, shot=None)
        return support_loader, query_loader


# =================================================================
# Main Entry Point
# =================================================================
if __name__ == "__main__":
    import argparse
    import sys

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.join(_this_dir, '..')
    sys.path.insert(0, _this_dir)
    sys.path.insert(0, _parent_dir)

    from train_incremental import IncrementalTrainer
    from models import DICAN_CBM
    from data.base_loader import DDRBaseDataset
    from data.fgadr_loader import FGADRSegDataset
    from utils.metrics import Evaluator

    # ─── Arguments ───
    parser = argparse.ArgumentParser(description='DICAN Training')
    parser.add_argument('--data_path', type=str, default='/root/DICAN_DATASETS')
    parser.add_argument('--epochs_base', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_tasks', type=int, default=4)
    parser.add_argument('--n_shot', type=int, default=10)
    parser.add_argument('--num_cluster', type=int, default=3)
    parser.add_argument('--adaptation_steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr_base', type=float, default=1e-4)
    parser.add_argument('--lr_inc', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.n_concepts = 4
    args.num_classes = 5

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*50}")
    print(f"🚀 DICAN Training (DDR+FGADR 통합)")
    print(f"   - Data Root: {args.data_path}")
    print(f"   - Device: {device}")
    print(f"   - Concepts: {args.n_concepts}")
    print(f"   - Clusters per concept: {args.num_cluster}")
    score_dim = args.n_concepts * 3
    print(f"   - Score dim: {score_dim} (Max+Mean+Std)")
    print(f"   - Head: {score_dim}→64→32→16→5 (Residual)")
    print(f"   - Tasks: {args.n_tasks} (1 base + {args.n_tasks-1} incremental)")
    print(f"   - Shot: {args.n_shot}")
    print(f"   - Adaptation steps: {args.adaptation_steps}")
    print(f"{'='*50}")

    # ─── 1. Data Loading ───
    print("\n[*] Loading Base Data...")
    ddr_root = os.path.join(args.data_path, 'DDR')
    fgadr_root = os.path.join(args.data_path, 'FGADR')
    print(f"    DDR:   {ddr_root}")
    print(f"    FGADR: {fgadr_root}")

    ddr_train = DDRBaseDataset(root_dir=ddr_root, split='train')
    ddr_val = DDRBaseDataset(root_dir=ddr_root, split='valid')
    fgadr_train = FGADRSegDataset(root_dir=fgadr_root, split='train')
    fgadr_val = FGADRSegDataset(root_dir=fgadr_root, split='valid')

    from torch.utils.data import ConcatDataset, DataLoader

    full_train = ConcatDataset([ddr_train, fgadr_train])
    full_val = ConcatDataset([ddr_val, fgadr_val])

    print(f"\n    ✅ Combined Dataset:")
    print(f"       Train: DDR({len(ddr_train)}) + FGADR({len(fgadr_train)}) = {len(full_train)}")
    print(f"       Valid: DDR({len(ddr_val)}) + FGADR({len(fgadr_val)}) = {len(full_val)}")

    nw = 4
    # FGADR-only (Dice 평가 전용)
    seg_train_loader = DataLoader(fgadr_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=nw, pin_memory=True)
    seg_val_loader = DataLoader(fgadr_val, batch_size=args.batch_size,
                                shuffle=False, num_workers=nw, pin_memory=True)
    # ★ DDR+FGADR 통합 (Phase 1-A/B/C 사용)
    full_train_loader = DataLoader(full_train, batch_size=args.batch_size,
                                   shuffle=True, num_workers=nw, pin_memory=True)
    full_val_loader = DataLoader(full_val, batch_size=args.batch_size,
                                 shuffle=False, num_workers=nw, pin_memory=True)

    loaders = {
        'seg_train': seg_train_loader,
        'seg_val': seg_val_loader,
        'full_train': full_train_loader,
        'full_val': full_val_loader,
    }

    # ─── 2. Model ───
    model = DICAN_CBM(
        num_concepts=args.n_concepts, num_classes=args.num_classes,
        feature_dim=2048, num_clusters=args.num_cluster
    ).to(device)

    # ─── 3. Base Training (Phase 1-A/B/C) ───
    base_trainer = BaseTrainer(args, model, device, loaders)
    model = base_trainer.run()

    # ─── 4. Incremental Setup ───
    inc_manager = IncrementalLoaderManager(
        data_root=args.data_path, n_shot=args.n_shot,
        batch_size=args.batch_size, seed=args.seed)

    evaluator = Evaluator(
        model=model, device=device, base_val_loader=full_val_loader,
        inc_loader_manager=inc_manager, args=args)

    # ─── 5. Base Evaluation ───
    print(f"\n[Eval] Evaluation after Base Session...")
    model.set_session_mode('eval')
    evaluator.evaluate_all_tasks(current_session_id=0)
    m0 = evaluator.calculate_metrics(current_session_id=0)
    print(f"   >>> Base Avg Acc: {m0['avg_acc']:.2f}%")
    print(f"   >>> Base Avg QWK: {m0['avg_kappa']:.4f}")

    # ─── 6. Incremental Tasks ───
    print(f"\n🔄 Starting Incremental Phase ({args.n_tasks - 1} tasks)")
    print(f"   Adaptation steps: {args.adaptation_steps}")

    inc_trainer = IncrementalTrainer(
        args=args, model=model, device=device, inc_loader=inc_manager)

    for task_id in range(1, args.n_tasks):
        task_acc = inc_trainer.train_task(task_id)
        
        model.set_session_mode('eval')
        evaluator.evaluate_all_tasks(current_session_id=task_id)
        m = evaluator.calculate_metrics(current_session_id=task_id)
        
        print(f"\n   📊 Metrics after Task {task_id}:")
        print(f"   - Avg Accuracy    : {m['avg_acc']:.2f}%")
        print(f"   - Avg QWK         : {m['avg_kappa']:.4f}")
        print(f"   - BWT             : {m['bwt']:.2f}%")
        print(f"   - BWT (QWK)       : {m['bwt_kappa']:.4f}")
        print(f"   - Forgetting      : {m['forgetting']:.2f}%")
        print(f"   - Forgetting (QWK): {m['forgetting_kappa']:.4f}")
        print(f"   - Task Accuracies : {m['raw_accs']}")
        print(f"   - Task QWKs       : {m['raw_kappas']}")

    # ─── 7. Final Summary ───
    final = evaluator.calculate_metrics(current_session_id=args.n_tasks - 1)
    print(f"\n{'='*50}")
    print(f"  🏁 DICAN Training Complete!")
    print(f"{'='*50}")
    print(f"   - Final Avg Acc : {final['avg_acc']:.2f}%")
    print(f"   - Final Avg QWK : {final['avg_kappa']:.4f}")
    print(f"   - Final BWT     : {final['bwt']:.2f}%")
    print(f"   - Final FWT     : {final['fwt']:.2f}%")
    print(f"   - Final Forget  : {final['forgetting']:.2f}%")
    print(f"\n   [QWK Details]")
    print(f"   - BWT (QWK)     : {final['bwt_kappa']:.4f}")
    print(f"   - FWT (QWK)     : {final['fwt_kappa']:.4f}")
    print(f"   - Forget (QWK)  : {final['forgetting_kappa']:.4f}")

    print(f"\n   [Accuracy Matrix R]")
    for i in range(args.n_tasks):
        print("   " + "  ".join(f"{evaluator.R[i][j]:6.2f}" for j in range(args.n_tasks)))

    print(f"\n   [QWK Matrix]")
    for i in range(args.n_tasks):
        print("   " + "  ".join(f"{evaluator.R_kappa[i][j]:6.4f}" for j in range(args.n_tasks)))