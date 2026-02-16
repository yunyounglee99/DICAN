import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# 프로젝트 모듈 임포트
from utils.loss import DICANLoss
from models.prototypes import PrototypeBank

class BaseTrainer:
    """
    [Phase 1: Base Training]
    - 학습 대상: Backbone, Head (Projector는 학습하지 않음!)
    - 구조: Backbone -> (Identity) -> Head
    - 목적: 충분한 데이터로 Backbone의 표현력을 극대화하고, 이를 바탕으로 초기 프로토타입 형성
    """
    def __init__(self, args, model, device, train_loader, val_loader):
        self.args = args
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # --------------------------------------------------------
        # 1. Projector Freeze (Base 단계에서는 Identity 취급)
        # --------------------------------------------------------
        self.model.backbone.train()
        if hasattr(self.model, 'head'):
            self.model.head.train()
            
        if hasattr(self.model, 'projector'):
            self.model.projector.eval() 
            for param in self.model.projector.parameters():
                param.requires_grad = False
        
        # --------------------------------------------------------
        # 2. Optimizer (Projector 파라미터 제외)
        # --------------------------------------------------------
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=self.args.lr_base, 
            weight_decay=self.args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs_base
        )

        # --------------------------------------------------------
        # 3. Loss & Prototype Bank
        # --------------------------------------------------------
        self.criterion = DICANLoss(
            mode='base',
            num_concepts=self.args.n_concepts, 
            num_classes=self.args.num_classes
        ).to(self.device)
        
        self.proto_bank = PrototypeBank(
            num_concepts=self.args.n_concepts, 
            feature_dim=2048 # ResNet50 기준
        ).to(self.device)

    def check_data_statistics(self):
        """학습 시작 전 데이터셋 크기 및 Shape 확인"""
        print(f"\n[{'='*10} Data Statistics (Base Session) {'='*10}]")
        train_len = len(self.train_loader.dataset)
        val_len = len(self.val_loader.dataset)
        print(f"  - Total Train Samples: {train_len}")
        print(f"  - Total Valid Samples: {val_len}")

        try:
            batch = next(iter(self.train_loader))
            # [수정] 딕셔너리 접근 방식 사용
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
                print(f"  - [Batch] Image Shape : {images.shape}")
                print(f"  - [Batch] Label Shape : {labels.shape}")
                if 'masks' in batch:
                    print(f"  - [Batch] Mask Shape  : {batch['masks'].shape}")
            else:
                print(f"  - [Batch] Data is not a dictionary: {type(batch)}")
        except Exception as e:
            print(f"  - [Warning] Failed to fetch batch for stats: {e}")
        print("="*45 + "\n")

    def train_epoch(self, epoch):
        self.model.train()
        if hasattr(self.model, 'projector'):
            self.model.projector.eval()
            
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Base Train Epoch {epoch+1}/{self.args.epochs_base}")
        
        for batch_idx, batch_data in enumerate(loop):
            # [핵심 수정] 리스트 언패킹 -> 딕셔너리 키 접근
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            # 마스크가 있는 경우만 로드 (Base Loader는 항상 있음)
            if 'masks' in batch_data:
                concept_masks = batch_data['masks'].to(self.device).float()
            else:
                raise ValueError('Segmentation masks are missing in Base Training!')
            
            self.optimizer.zero_grad()
            
            # Forward (Projector OFF)
            model_outputs = self.model(images, masks=concept_masks)

            targets = {'label': labels, 'masks': concept_masks}
            
            # Loss 계산
            loss, log_dict = self.criterion(model_outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            logits = model_outputs['logits']
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"})
            
        self.scheduler.step()
        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # [핵심 수정] 딕셔너리 접근
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                if 'masks' in batch_data:
                    concept_masks = batch_data['masks'].to(self.device).float()
                else:
                    concept_masks = torch.zeros(images.size(0), self.args.n_concepts, 224, 224).to(self.device)

                # Validation에서도 Projector 없이 평가
                model_outputs = self.model(images, masks = concept_masks)
                targets = {'label': labels, 'masks': concept_masks}
                
                loss, log_dict = self.criterion(model_outputs, targets)
                val_loss += loss.item()

                logits = model_outputs['logits']
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        avg_loss = val_loss / len(self.val_loader)
        print(f"[*] Base Validation - Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        return acc

    def extract_and_save_prototypes(self):
        """
        Base 학습 완료 후, Prototype 추출
        (use_projector=False로 추출해야 Backbone의 순수 Feature가 저장됨)
        """
        print("\n[*] Extracting Base Prototypes (Backbone Features)...")
        self.model.eval()
        
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.train_loader, desc="Extracting Features"):
                # [핵심 수정] 딕셔너리 접근
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                # Base Session에서는 Mask를 활용해 Prototype 업데이트
                if 'masks' in batch_data:
                    masks = batch_data['masks'].to(self.device).float()
                else:
                    continue # 마스크 없으면 스킵
                
                # Projector 없이 Feature 추출
                _, features = self.model(images, use_projector=False)
                
                # 배치 단위로 프로토타입 업데이트 (PrototypeBank 내부에서 처리)
                self.proto_bank.update_with_masks(features, masks)
        
        # 모델에 프로토타입 주입
        if hasattr(self.model, 'prototypes'):
            self.model.prototypes = self.proto_bank.prototypes.clone()
            
        print(f"[*] Base Prototypes Updated via Masked Averaging.")

    def run(self):
        print(f"\n{'='*20} [Phase 1] Base Training Start (Projector OFF) {'='*20}")
        self.check_data_statistics()
        
        best_acc = 0.0
        for epoch in range(self.args.epochs_base):
            self.train_epoch(epoch)
            
            if (epoch + 1) % 1 == 0:
                val_acc = self.validate()
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_model("best_base_model.pth")
        
        print("[*] Loading best model for prototype extraction...")
        self.load_model("best_base_model.pth")
        self.extract_and_save_prototypes()
        
        print(f"{'='*20} Base Training Finished {'='*20}\n")
        return self.model

    def save_model(self, filename):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.args.save_path, filename))

    def load_model(self, filename):
        path = os.path.join(self.args.save_path, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))