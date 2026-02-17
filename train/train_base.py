"""
DICAN Base Training - 3-Phase (Seg-Supervised)
=================================================
Phase 1-A: Backbone + TempHead + SegHead 학습
           - Classification Loss: GAP → TempHead → CE (등급 분류)
           - Segmentation Loss:   SegHead → BCE with mask (병변 위치)
           ★ Backbone이 "어디에 병변이 있는지"를 학습함
           
Phase 1-B: Backbone Freeze → Masked Pooling → Prototype 추출
           ★ 병변 인식 feature로부터 고품질 prototype 구축

Phase 1-C: Backbone + Prototype Freeze → CBM Head만 학습
           ★ concept_scores → DR Grade 매핑 학습
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
                for k, name in enumerate(["EX", "HE", "MA", "SE"]):
                    active = (masks[:, k].sum(dim=(1,2)) > 0).sum().item()
                    total_px = (masks[:, k] > 0).sum().item()
                    print(f"    {name}: {active}/{masks.size(0)} images active, "
                          f"{total_px} total pixels")
        except Exception as e:
            print(f"  [Warning] {e}")
        print("=" * 35 + "\n")

    # =================================================================
    # Phase 1-A: Backbone Pre-training (Classification + Segmentation)
    # =================================================================
    def phase_1a_pretrain(self):
        """
        [핵심]
        두 가지 Loss로 Backbone 학습:
        
        1) L_cls = CrossEntropy(TempHead(GAP(features)), label)
           → "이 이미지가 Grade 몇인지" 학습
           
        2) L_seg = BCE(SegHead(features), mask_resized)
           → "7×7 feature map의 각 위치에 어떤 병변이 있는지" 학습
           
        Backbone이 받는 gradient:
           ∂L_cls/∂θ_backbone : 전역적 분류 능력
           ∂L_seg/∂θ_backbone : 공간적 병변 인식 능력
           
        → 두 gradient가 합쳐져서 Backbone은
          "병변 위치를 알면서 등급도 구분하는" feature를 생성하게 됨
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-A: Backbone Pretrain (Cls + Seg Supervision)")
        print(f"{'='*60}")
        
        self.model.set_session_mode('pretrain')
        
        trainable_params = self.model.get_trainable_params()
        n_params = sum(p.numel() for p in trainable_params)
        print(f"  Trainable parameters: {n_params:,}")
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr_base,
            weight_decay=self.args.weight_decay
        )
        
        epochs = self.args.epochs_base
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 클래스 가중치 (DR 불균형 대응)
        class_weights = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0], device=self.device)
        
        # Seg Loss 가중치 (너무 크면 Backbone이 분류를 잊고 세그에만 집중)
        lambda_seg = 1.0
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            # Projector, CBM Head는 eval 유지
            self.model.projector.eval()
            self.model.head.eval()
            
            total_loss = 0.0
            total_cls_loss = 0.0
            total_seg_loss = 0.0
            correct = 0
            total = 0
            
            loop = tqdm(self.train_loader, 
                       desc=f"[1-A] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                cls_logits = outputs['logits']    # [B, 5]
                seg_pred = outputs['seg_pred']    # [B, 4, 7, 7]
                
                # ----- (a) Classification Loss -----
                loss_cls = F.cross_entropy(cls_logits, labels, weight=class_weights)
                
                # ----- (b) Segmentation Loss -----
                # 마스크를 feature map 크기(7×7)로 리사이징
                # Nearest: 작은 MA도 보존 (README의 Dr.DR Tip 반영)
                masks_resized = F.interpolate(
                    masks, 
                    size=seg_pred.shape[2:],  # (7, 7)
                    mode='nearest'
                )  # [B, 4, 7, 7]
                
                # Sigmoid + BCE: 각 위치별, 각 병변별 독립적으로 존재 여부 학습
                loss_seg = F.binary_cross_entropy_with_logits(
                    seg_pred, masks_resized
                )
                
                # ----- Total Loss -----
                loss = loss_cls + lambda_seg * loss_seg
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_cls_loss += loss_cls.item()
                total_seg_loss += loss_seg.item()
                
                _, predicted = cls_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                loop.set_postfix({
                    "cls": f"{loss_cls.item():.3f}",
                    "seg": f"{loss_seg.item():.3f}",
                    "acc": f"{100.*correct/total:.1f}%"
                })
            
            scheduler.step()
            train_acc = 100. * correct / total
            avg_seg = total_seg_loss / len(self.train_loader)
            
            # Validate
            val_acc = self._validate_pretrain(class_weights)
            
            print(f"  Epoch {epoch+1}: TrainAcc={train_acc:.1f}%, ValAcc={val_acc:.1f}%, "
                  f"SegLoss={avg_seg:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("phase1a_best.pth")
                print(f"  ★ Best: {val_acc:.2f}%")
        
        self.load_model("phase1a_best.pth")
        
        # Seg Head 품질 확인
        self._verify_seg_quality()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}%")
        return best_val_acc

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

    def _verify_seg_quality(self):
        """
        Phase 1-A 학습 후, SegHead가 실제로 병변 위치를 잡는지 검증.
        마스크가 있는 샘플에서 SegHead 출력과 Ground Truth의 IoU/Dice를 측정.
        """
        print("\n[*] Verifying Segmentation Quality...")
        self.model.eval()
        
        concept_names = ["EX", "HE", "MA", "SE"]
        concept_dice = {k: [] for k in range(4)}
        
        with torch.no_grad():
            count = 0
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                outputs = self.model(images)
                seg_pred = outputs['seg_pred']  # [B, 4, 7, 7]
                
                masks_resized = F.interpolate(
                    masks, size=seg_pred.shape[2:], mode='nearest'
                )
                
                # Sigmoid → 이진화
                seg_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                
                for k in range(4):
                    pred_k = seg_binary[:, k]
                    gt_k = masks_resized[:, k]
                    
                    # 마스크가 있는 샘플만
                    has_lesion = (gt_k.sum(dim=(1,2)) > 0)
                    if has_lesion.any():
                        intersection = (pred_k[has_lesion] * gt_k[has_lesion]).sum()
                        union = pred_k[has_lesion].sum() + gt_k[has_lesion].sum()
                        if union > 0:
                            dice = (2 * intersection / union).item()
                            concept_dice[k].append(dice)
                
                count += 1
                if count >= 20:  # 20 배치만 샘플링
                    break
        
        print(f"  Segmentation Dice Scores (7×7 resolution):")
        for k, name in enumerate(concept_names):
            if concept_dice[k]:
                avg_dice = np.mean(concept_dice[k])
                print(f"    {name}: {avg_dice:.4f} (from {len(concept_dice[k])} batches)")
            else:
                print(f"    {name}: N/A (no positive samples in validation)")
        
        print("  → Dice > 0.3 이면 Backbone이 병변 위치를 인식하고 있음")
        print("  → Dice < 0.1 이면 lambda_seg를 올리거나 epoch을 늘려야 함\n")

    # =================================================================
    # Phase 1-B: Prototype Extraction (Backbone Frozen)
    # =================================================================
    def phase_1b_extract_prototypes(self):
        """
        [핵심]
        Phase 1-A에서 Backbone이 병변 위치를 학습했으므로:
        - feature map의 "출혈 위치" 픽셀은 출혈 관련 activation이 높음
        - feature map의 "배경" 픽셀은 출혈 관련 activation이 낮음
        
        여기서 Masked GAP를 하면:
        - 마스크가 가리키는 픽셀들의 feature만 평균
        - = "출혈 관련 activation이 높은 픽셀들"의 평균
        - = 순도 높은 "출혈 prototype"
        
        만약 Phase 1-A에서 Seg Loss 없이 학습했다면:
        - feature map의 각 픽셀이 "여기가 출혈인지" 모름
        - Masked GAP해도 "그냥 아무 feature의 평균" → 저품질 prototype
        """
        print(f"\n{'='*60}")
        print(f"  Phase 1-B: Prototype Extraction (Backbone Frozen)")
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
    # Phase 1-C: Head Training (CBM Logic)
    # =================================================================
    def phase_1c_train_head(self):
        """
        [구조]
        Backbone(고정, 병변 인식) → Prototype(고정) → Hybrid Score [B,8] → Head(학습)
        
        Head가 학습하는 것:
        - "EX_max가 높고 HE_mean이 중간이면 Grade 2다"
        - "모든 max가 낮으면 Grade 0이다"
        - "SE_max가 높으면 Grade 3 이상이다"
        
        Backbone이 이미 병변 위치를 인식하므로:
        - Prototype과의 similarity가 실제 병변 유무를 잘 반영
        - concept_scores의 변별력이 높아 Head 학습이 용이
        """
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
        
        for epoch in range(epochs):
            self.model.head.train()
            total_loss = 0.0
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
                
                # Sparsity: Grade 0은 모든 concept 비활성화
                loss_sp = torch.tensor(0.0, device=self.device)
                normal = (labels == 0)
                if normal.any() and concept_scores is not None:
                    max_scores = concept_scores[normal, :self.model.num_concepts]
                    loss_sp = torch.relu(max_scores).mean()
                
                loss = loss_cls + 0.3 * loss_sp
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
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
            print(f"  Epoch {epoch+1}: Train={100.*correct/total:.1f}%, Val={val_acc:.1f}%, "
                  f"Scale={self.model.prototypes.logit_scale.exp().item():.1f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("phase1c_best.pth")
                print(f"  ★ Best: {val_acc:.2f}%")
        
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
        """Grade별 Concept Score 분포 분석"""
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
                row = f"  {g:>5} | " + " | ".join(f"{m.item():>7.2f}" for m in means)
                print(row)
        
        print("\n  → Grade 3,4의 Max scores가 Grade 0보다 확연히 높아야 정상")
        print("  → 차이가 작으면 Prototype 품질 문제 (Phase 1-A seg loss 확인)\n")

    # =================================================================
    # Main Run
    # =================================================================
    def run(self):
        print(f"\n{'='*60}")
        print(f"  DICAN 3-Phase Base Training")
        print(f"{'='*60}")
        self.check_data_statistics()
        
        acc_1a = self.phase_1a_pretrain()
        self.phase_1b_extract_prototypes()
        acc_1c = self.phase_1c_train_head()
        
        print(f"\n{'='*60}")
        print(f"  Base Training Complete!")
        print(f"  Phase 1-A (Backbone+Seg): {acc_1a:.2f}%")
        print(f"  Phase 1-C (CBM Head):     {acc_1c:.2f}%")
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