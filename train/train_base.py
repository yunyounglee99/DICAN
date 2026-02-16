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
        # [핵심 수정 1] Projector Freeze (Base 단계에서는 Identity 취급)
        # --------------------------------------------------------
        # Backbone과 Head는 학습 모드
        self.model.backbone.train()
        if hasattr(self.model, 'head'):
            self.model.head.train()
            
        # Projector는 평가 모드 & Gradient 계산 끄기
        if hasattr(self.model, 'projector'):
            self.model.projector.eval() 
            for param in self.model.projector.parameters():
                param.requires_grad = False
        
        # --------------------------------------------------------
        # [핵심 수정 2] Optimizer에 Projector 파라미터 제외
        # --------------------------------------------------------
        # requires_grad가 True인 파라미터(Backbone, Head)만 필터링하여 전달
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=self.args.lr_base, 
            weight_decay=self.args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs_base
        )

        # Loss Function & Prototype Bank
        self.criterion = DICANLoss(args) 
        self.proto_bank = PrototypeBank(args, device)

    def train_epoch(self, epoch):
        self.model.train()
        # Projector는 확실히 eval 모드로 고정 (혹시 model.train() 호출로 풀릴까봐 안전장치)
        if hasattr(self.model, 'projector'):
            self.model.projector.eval()
            
        total_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc=f"Base Train Epoch {epoch+1}/{self.args.epochs_base}")
        
        for batch_idx, batch_data in enumerate(loop):
            if len(batch_data) == 3:
                images, labels, concept_masks = batch_data
                concept_masks = concept_masks.to(self.device).float()
            else:
                images, labels = batch_data
                concept_masks = None

            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # --------------------------------------------------------
            # [핵심 수정 3] Forward 시 Projector 사용 안 함 (Identity)
            # --------------------------------------------------------
            # use_projector=False를 전달하여 backbone feature가 바로 concept이 되도록 유도
            logits, concepts = self.model(images, use_projector=False)
            
            loss_dict = self.criterion(logits, labels, concepts, concept_masks)
            loss = loss_dict['total']
            
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
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
                if len(batch_data) == 3:
                    images, labels, concept_masks = batch_data
                    concept_masks = concept_masks.to(self.device).float()
                else:
                    images, labels = batch_data
                    concept_masks = None

                images, labels = images.to(self.device), labels.to(self.device)
                
                # Validation에서도 Projector 없이 평가
                logits, concepts = self.model(images, use_projector=False)
                
                loss_dict = self.criterion(logits, labels, concepts, concept_masks)
                val_loss += loss_dict['total'].item()
                
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
        이때도 use_projector=False로 추출해야 Backbone의 순수 Feature가 저장됨
        """
        print("\n[*] Extracting Base Prototypes (Backbone Features)...")
        self.model.eval()
        
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.train_loader, desc="Extracting Features"):
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                else:
                    images, labels = batch_data
                
                images = images.to(self.device)
                
                # Projector 없이(Identity) Feature 추출
                _, concepts = self.model(images, use_projector=False)
                
                all_concepts.append(concepts.cpu())
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_concepts, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        self.proto_bank.update(features, labels, task_id=0)
        
        if hasattr(self.model, 'prototypes'):
            self.model.prototypes = self.proto_bank.get_prototypes()
            
        print(f"[*] Base Prototypes Saved. Classes: {len(torch.unique(labels))}")

    def run(self):
        print(f"\n{'='*20} [Phase 1] Base Training Start (Projector OFF) {'='*20}")
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