import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# 프로젝트 모듈 임포트
from utils.loss import DICANLoss
from models.prototypes import PrototypeBank

class IncrementalTrainer:
    """
    DICAN 모델의 Incremental Task (Phase 2) 트레이너
    - Adaptation: Few-shot 데이터를 이용해 'Concept Projector'만 미세조정 (Backbone Freeze)
    - Registration: PrototypeBank 업데이트
    - Evaluation: Nearest Prototype Classification
    """
    def __init__(self, args, model, device, inc_loader):
        self.args = args
        self.model = model
        self.device = device
        self.inc_loader = inc_loader
        
        # Prototype Bank 초기화
        self.proto_bank = PrototypeBank(args, device)
        
        # 모델에 기존 프로토타입이 있다면 로드 (연속 학습을 위해)
        if hasattr(self.model, 'prototypes') and self.model.prototypes is not None:
            self.proto_bank.prototypes = self.model.prototypes.clone()

        # Loss Function (Adaptation용)
        self.criterion = DICANLoss(args)

    def train_task(self, task_id):
        print(f"\n{'='*20} [Phase 2] Incremental Task {task_id} Start {'='*20}")
        
        # 데이터 로드
        support_loader, query_loader = self.inc_loader.get_incremental_loaders(task_id)
        
        # 1. Adaptation (Fine-tuning)
        # [중요] Backbone과 Head는 얼리고 Projector만 학습합니다.
        if self.args.adaptation_steps > 0:
            self.adapt_to_domain(support_loader, task_id)
            
        # 2. Prototype Registration
        self.register_prototypes(support_loader, task_id)
        
        # 3. Evaluation
        acc = self.evaluate(query_loader, task_id)
        
        print(f"{'='*20} Incremental Task {task_id} Finished (Acc: {acc:.2f}%) {'='*20}\n")
        return acc

    def adapt_to_domain(self, support_loader, task_id):
        """
        Support Set을 사용하여 Concept Projector만 미세조정(Fine-tuning)합니다.
        Backbone과 Head는 Freeze하여 지식 망각(Forgetting)을 방지합니다.
        """
        print(f"[*] Adapting to Task {task_id} (Projector Only Fine-tuning)...")
        
        # -----------------------------------------------------------
        # [핵심 수정] Freeze Logic 적용
        # -----------------------------------------------------------
        # 1. Backbone Freeze (평가 모드 + Gradient 계산 끔)
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        # 2. Head Freeze (평가 모드 + Gradient 계산 끔)
        if hasattr(self.model, 'head') and self.model.head is not None:
            self.model.head.eval()
            for param in self.model.head.parameters():
                param.requires_grad = False
                
        # 3. Projector Unfreeze (학습 모드 + Gradient 계산 켬)
        self.model.projector.train()
        for param in self.model.projector.parameters():
            param.requires_grad = True
            
        # 4. Optimizer에는 Projector 파라미터만 전달!
        optimizer = optim.SGD(
            self.model.projector.parameters(), # <--- Projector만 전달
            lr=self.args.lr_inc, 
            momentum=0.9
        )
        
        iterator = iter(support_loader)
        
        for step in range(self.args.adaptation_steps):
            try:
                batch_data = next(iterator)
            except StopIteration:
                iterator = iter(support_loader)
                batch_data = next(iterator)
            
            # 데이터 로더 반환값 처리 (Mask 유무)
            if len(batch_data) == 3:
                images, labels, concept_masks = batch_data
                concept_masks = concept_masks.to(self.device).float()
            else:
                images, labels = batch_data
                concept_masks = None
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            logits, concepts = self.model(images)
            
            # Loss Calculation (Mask 유무에 따라 처리)
            loss_dict = self.criterion(logits, labels, concepts, concept_masks)
            loss = loss_dict['total']
            
            loss.backward()
            optimizer.step()
            
            # 학습 상황 로깅 (옵션)
            if (step + 1) % 10 == 0:
                print(f"    Step [{step+1}/{self.args.adaptation_steps}] Loss: {loss.item():.4f}")

        # 학습 종료 후 모델 전체를 다시 Eval 모드로 전환 (안전장치)
        self.model.eval()

    def register_prototypes(self, support_loader, task_id):
        """새로운 태스크의 데이터로 프로토타입 갱신/등록"""
        print(f"[*] Registering Prototypes for Task {task_id}...")
        self.model.eval()
        
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in support_loader:
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                else:
                    images, labels = batch_data
                    
                images = images.to(self.device)
                
                # Concept Feature 추출
                _, concepts = self.model(images)
                
                all_concepts.append(concepts.cpu())
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_concepts, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # PrototypeBank에 업데이트
        self.proto_bank.update(features, labels, task_id)
        
        # 모델에 반영
        if hasattr(self.model, 'prototypes'):
            self.model.prototypes = self.proto_bank.get_prototypes()

    def evaluate(self, query_loader, task_id):
        """Query Set 평가 (프로토타입 기반 분류)"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in tqdm(query_loader, desc=f"Evaluating Task {task_id}"):
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                else:
                    images, labels = batch_data

                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward Pass (logits는 프로토타입과의 거리 기반 점수)
                logits, _ = self.model(images)
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return acc