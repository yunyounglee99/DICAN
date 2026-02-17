"""
DICAN Base Training - 3-Phase (Pixel-Level Seg Supervision)
=============================================================
Phase 1-A: Backbone + TempHead + SegDecoder 학습
           - L_cls: GAP → TempHead → CE (등급 분류)
           - L_seg: SegDecoder → [B,4,224,224] → BCE with 원본 마스크 ★
           Backbone이 pixel 단위로 "여기에 MA가 있다"를 학습

Phase 1-B: Backbone Freeze → Masked GAP → Prototype 추출

Phase 1-C: Backbone + Prototype Freeze → CBM Head 학습
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
                    active_imgs = (masks[:, k].sum(dim=(1,2)) > 0).sum().item()
                    active_pixels = (masks[:, k] > 0).sum().item()
                    total_pixels = masks[:, k].numel()
                    ratio = 100.0 * active_pixels / total_pixels
                    print(f"    {name}: {active_imgs}/{masks.size(0)} images, "
                          f"{active_pixels:,} pixels ({ratio:.2f}%)")
        except Exception as e:
            print(f"  [Warning] {e}")
        print("=" * 35 + "\n")

    # =================================================================
    # Phase 1-A: Backbone + Pixel-Level Segmentation
    # =================================================================
    def phase_1a_pretrain(self):
        """
        [핵심 차이: 7×7 vs 224×224]
        
        기존 (7×7):
          SegHead([2048,7,7]) → [4,7,7]
          mask를 7×7로 축소 → BCE
          → 32×32 블록 단위로만 병변 유무 판단
          → MA(수 픽셀) 완전 소실
        
        수정 (224×224):
          SegDecoder(multi_scale) → [4,224,224]
          원본 mask [4,224,224]과 직접 BCE
          → 각 pixel에서 병변 유무 판단
          → MA도 pixel 단위로 학습 가능
          → Backbone의 layer1(56×56)까지 gradient 전파
            → 고해상도 레벨에서도 병변 경계 인식
        """
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
        
        # Seg Loss 가중치
        # 너무 크면 분류를 희생하고 세그에만 집중
        # 너무 작으면 Backbone이 위치를 못 배움
        lambda_seg = 1.0
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            self.model.projector.eval()
            self.model.head.eval()
            
            total_loss = 0.0
            total_cls = 0.0
            total_seg = 0.0
            correct = 0
            total = 0
            
            loop = tqdm(self.train_loader, desc=f"[1-A] Epoch {epoch+1}/{epochs}")
            
            for batch_data in loop:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()  # [B, 4, 224, 224]
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                cls_logits = outputs['logits']      # [B, 5]
                seg_pred = outputs['seg_pred']       # [B, 4, 224, 224] ★ pixel-level
                
                # ─── Classification Loss ───
                loss_cls = F.cross_entropy(cls_logits, labels, weight=class_weights)
                
                # ─── Pixel-Level Segmentation Loss ───
                # seg_pred: [B, 4, 224, 224] (logits)
                # masks:    [B, 4, 224, 224] (binary: 0 or 1)
                # 마스크를 리사이징할 필요 없음! 같은 해상도에서 직접 비교!
                #
                # Focal Loss 적용: 병변 pixel이 전체의 1~5%이므로
                # 쉬운 배경 pixel의 loss를 줄이고 어려운 병변 pixel에 집중
                loss_seg = self._focal_bce_loss(seg_pred, masks, gamma=2.0, alpha=0.75)
                
                loss = loss_cls + lambda_seg * loss_seg
                
                loss.backward()
                
                # Gradient Clipping (Decoder가 커서 gradient 폭발 방지)
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
                    "seg": f"{loss_seg.item():.3f}",
                    "acc": f"{100.*correct/total:.1f}%"
                })
            
            scheduler.step()
            train_acc = 100. * correct / total
            avg_seg = total_seg / len(self.train_loader)
            
            val_acc = self._validate_pretrain(class_weights)
            
            print(f"  Epoch {epoch+1}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, "
                  f"SegLoss={avg_seg:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("phase1a_best.pth")
                print(f"  ★ Best: {val_acc:.2f}%")
        
        self.load_model("phase1a_best.pth")
        self._verify_seg_quality_pixel_level()
        
        print(f"\n[1-A] Complete. Best Val Acc: {best_val_acc:.2f}%")
        return best_val_acc

    def _focal_bce_loss(self, pred, target, gamma=2.0, alpha=0.75):
        """
        Focal Loss for Binary Segmentation
        
        [왜 Focal Loss인가?]
        DR 이미지에서 병변 pixel은 전체의 1~5%.
        일반 BCE를 쓰면 95%의 배경 pixel이 loss를 지배하여
        모델이 "전부 0으로 예측"하는 것이 가장 쉬운 해법이 됨.
        
        Focal Loss는:
        - 이미 잘 맞추는 배경 pixel의 loss를 γ 제곱으로 줄이고
        - 틀리기 쉬운 병변 pixel의 loss는 유지
        → 병변 경계(어려운 pixel)에 집중
        
        alpha: 양성(병변) pixel에 대한 가중치 (0.75 = 병변에 3배 가중)
        gamma: focusing parameter (2.0이 표준)
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)  # 맞힌 확률
        
        # Focal weight: 쉬운 sample은 (1-pt)^γ → 0에 가까워짐
        focal_weight = (1.0 - pt) ** gamma
        
        # Alpha weight: 양성 pixel에 더 큰 가중치
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
        224×224 해상도에서 Segmentation 품질 검증
        Dice Score로 각 병변별 검출 성능 측정
        """
        print("\n[*] Pixel-Level Segmentation Quality (224×224):")
        self.model.eval()
        self.model.set_session_mode('pretrain')  # SegDecoder 접근 필요
        self.model.eval()  # 다시 eval
        
        concept_names = ["EX", "HE", "MA", "SE"]
        dice_scores = {k: [] for k in range(4)}
        iou_scores = {k: [] for k in range(4)}
        
        with torch.no_grad():
            count = 0
            for batch_data in self.val_loader:
                images = batch_data['image'].to(self.device)
                masks = batch_data['masks'].to(self.device).float()
                
                # Multi-scale forward
                feat4, multi_scale = self.model.backbone(images, return_multi_scale=True)
                seg_pred = self.model.seg_decoder(multi_scale)  # [B, 4, 224, 224]
                seg_binary = (torch.sigmoid(seg_pred) > 0.5).float()
                
                for k in range(4):
                    pred_k = seg_binary[:, k]   # [B, 224, 224]
                    gt_k = masks[:, k]           # [B, 224, 224]
                    
                    # 병변이 있는 이미지만
                    has_lesion = (gt_k.sum(dim=(1,2)) > 0)
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
                if count >= 30:
                    break
        
        print(f"  {'Concept':>8} | {'Dice':>8} | {'IoU':>8} | {'Samples':>8}")
        print(f"  {'-'*40}")
        for k, name in enumerate(concept_names):
            if dice_scores[k]:
                d = np.mean(dice_scores[k])
                i = np.mean(iou_scores[k])
                n = len(dice_scores[k])
                print(f"  {name:>8} | {d:>8.4f} | {i:>8.4f} | {n:>8}")
            else:
                print(f"  {name:>8} | {'N/A':>8} | {'N/A':>8} | {'0':>8}")
        
        print(f"\n  해석 기준:")
        print(f"    Dice > 0.5: 병변 위치를 잘 인식함 (Good)")
        print(f"    Dice 0.3~0.5: 대략적 위치는 알지만 경계가 부정확 (OK)")
        print(f"    Dice < 0.3: 병변 인식이 부족 → lambda_seg 증가 필요")
        print(f"    MA의 Dice가 낮은 것은 정상 (극소 병변)")
        print()

    # =================================================================
    # Phase 1-B: Prototype Extraction
    # =================================================================
    def phase_1b_extract_prototypes(self):
        """
        Backbone이 pixel-level seg 학습을 거쳤으므로:
        - feature map [2048, 7, 7]의 각 위치가 해당 영역의 병변 특성을 인코딩
        - Masked GAP 시 마스크 영역의 feature = 병변 고유 특징 벡터
        - Prototype 품질이 7×7 seg 학습 대비 훨씬 향상됨
        
        특히 MA:
        - 7×7에서는 MA pixel이 사라져서 prototype이 부정확했음
        - 224×224에서 학습했으므로 Backbone의 layer1~3이
          MA의 미세한 특징까지 인코딩하고 있고,
          이것이 layer4의 7×7까지 전파되어 있음
        """
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
    # Phase 1-C: Head Training
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
                
                # Sparsity
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
        print(f"  DICAN 3-Phase Training (Pixel-Level Seg)")
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