import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 프로젝트 모듈 임포트
from utils.loss import TaskLoss, ConceptLoss  # utils/loss.py에 정의된 클래스 사용
from models.prototypes import PrototypeManager
from utils.args import ArgumentParser # args 관리를 위한 유틸리티가 있다면 사용, 없으면 main에서 전달받음

class BaseTrainer:
    """
    DICAN 모델의 Base Task (초기 학습)를 담당하는 트레이너 클래스입니다.
    Backbone과 Concept Layer를 학습시키고, 초기 프로토타입을 형성합니다.
    """
    def __init__(self, args, model, device, train_loader, val_loader):
        self.args = args
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # --------------------------------------------------------
        # 1. Optimizer & Scheduler 설정
        # --------------------------------------------------------
        # Base 학습 때는 Backbone과 Concept Layer, Head 모두를 충분히 학습시켜야 합니다.
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr_base, 
            weight_decay=self.args.weight_decay
        )
        
        # Cosine Annealing 스케줄러 사용 (학습 후반부 안정화)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs_base
        )

        # --------------------------------------------------------
        # 2. Loss Function 설정 (utils/loss.py 활용)
        # --------------------------------------------------------
        # Task Loss: Grading(분류) 정확도를 높이기 위함 (CrossEntropy 등)
        self.criterion_task = TaskLoss() 
        
        # Concept Loss: Mask를 통해 모델이 올바른 Concept을 학습하도록 유도 (BCE 등)
        # weight는 Concept Loss의 비중을 조절합니다 (예: lambda_c)
        self.criterion_concept = ConceptLoss() 
        self.lambda_c = getattr(self.args, 'lambda_c', 1.0) # args에 없으면 기본값 1.0

        # Prototype Manager 초기화
        self.prototype_manager = PrototypeManager(self.args, self.device)

    def train_epoch(self, epoch):
        """한 Epoch 동안의 학습 루프"""
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_concept_loss = 0.0
        correct = 0
        total = 0
        
        # Progress Bar 설정
        loop = tqdm(self.train_loader, desc=f"Base Train Epoch {epoch+1}/{self.args.epochs_base}")
        
        for batch_idx, (images, labels, concept_masks) in enumerate(loop):
            # 데이터 로더에서 (이미지, 라벨, 컨셉 마스크)를 반환한다고 가정
            images = images.to(self.device)
            labels = labels.to(self.device)
            concept_masks = concept_masks.to(self.device).float() # Mask는 float 형태여야 함
            
            self.optimizer.zero_grad()
            
            # ----------------------------------------------------
            # Forward Pass (DICAN_CBM)
            # ----------------------------------------------------
            # 모델은 최종 예측값(logits)과 내부 Concept 활성값(concepts)을 반환해야 함
            logits, concepts = self.model(images)
            
            # ----------------------------------------------------
            # Loss Calculation
            # ----------------------------------------------------
            # 1. Grading (Task) Loss
            loss_t = self.criterion_task(logits, labels)
            
            # 2. Concept Alignment Loss (Mask 기반 학습)
            # 예측된 concept과 실제 유효한 concept mask 사이의 차이를 줄임
            loss_c = self.criterion_concept(concepts, concept_masks)
            
            # 3. Total Loss
            loss = loss_t + (self.lambda_c * loss_c)
            
            # Backward Pass
            loss.backward()
            self.optimizer.step()
            
            # ----------------------------------------------------
            # Metrics Logging
            # ----------------------------------------------------
            total_loss += loss.item()
            total_task_loss += loss_t.item()
            total_concept_loss += loss_c.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # tqdm 업데이트
            loop.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "TaskLoss": f"{loss_t.item():.4f}",
                "ConLoss": f"{loss_c.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })
            
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc

    def validate(self):
        """검증 데이터셋 평가"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, concept_masks in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(images) # Validation에서는 Concept Loss 계산 생략 가능
                
                loss = self.criterion_task(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        avg_loss = val_loss / len(self.val_loader)
        
        print(f"[*] Validation - Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        return acc

    def extract_and_save_prototypes(self):
        """
        [핵심 연구 구현]
        Base Training이 끝난 후, 학습된 Backbone+Concept Layer를 통과한
        데이터들의 'Concept Feature' 평균(Prototype)을 계산하여 저장합니다.
        이는 이후 Incremental Learning에서 Replay 없이 지식을 유지하는 핵심입니다.
        """
        print("\n[*] Extracting Base Prototypes using trained concepts...")
        self.model.eval()
        
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.train_loader, desc="Extracting Features"):
                images = images.to(self.device)
                
                # 모델에서 'concepts' (bottleneck features)를 추출
                # forward 시 logits 뿐만 아니라 concepts를 반환받음
                _, concepts = self.model(images)
                
                # Concept Activation 값을 저장
                all_concepts.append(concepts.cpu())
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_concepts, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # PrototypeManager에 위임하여 클래스별 평균(Prototype) 계산 및 저장
        # task_id=0 은 Base Task를 의미
        self.prototype_manager.update_prototypes(features, labels, task_id=0)
        
        # 모델 내부의 프로토타입 버퍼도 업데이트 (Inference 시 사용)
        self.model.prototypes = self.prototype_manager.get_prototypes()
        print(f"[*] Base Prototypes Saved. Total Classes: {len(torch.unique(labels))}")

    def run(self):
        """전체 학습 프로세스 실행"""
        print(f"\n{'='*20} [Phase 1] Base Training Start {'='*20}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.args.epochs_base}")
        
        best_acc = 0.0
        
        for epoch in range(self.args.epochs_base):
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation 수행 (매 epoch 혹은 특정 주기마다)
            if (epoch + 1) % 1 == 0: # 매 epoch마다 확인
                val_acc = self.validate()
                
                # 최고 성능 모델 저장
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_model("best_base_model.pth")
                    print(f"[*] Best Model Saved at Epoch {epoch+1} (Acc: {val_acc:.2f}%)")
        
        # 최적 모델 로드 (Prototype 추출을 위해 가장 성능 좋은 모델 사용)
        print("[*] Loading best model for prototype extraction...")
        self.load_model("best_base_model.pth")
        
        # Prototype 추출 및 저장
        self.extract_and_save_prototypes()
        
        print(f"{'='*20} Base Training Finished {'='*20}\n")
        return self.model

    def save_model(self, filename):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        save_path = os.path.join(self.args.save_path, filename)
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, filename):
        save_path = os.path.join(self.args.save_path, filename)
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))