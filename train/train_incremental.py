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
        
        # [Fix] Prototype Bank 초기화 수정 (정확한 인자 전달)
        self.proto_bank = PrototypeBank(
            num_concepts=self.args.n_concepts, 
            feature_dim=2048 # ResNet50 기준
        ).to(self.device)
        
        # 모델에 기존 프로토타입이 있다면 로드 (연속 학습을 위해)
        if hasattr(self.model, 'prototypes') and self.model.prototypes is not None:
            self.proto_bank.prototypes = self.model.prototypes.clone()
            if hasattr(self.proto_bank, 'initialized'):
                self.proto_bank.initialized.fill_(True)

        # Loss Function (Adaptation용)
        self.criterion = DICANLoss(
            mode='incremental',
            num_concepts=self.args.n_concepts,
            num_classes=self.args.num_classes
        ).to(self.device)

    def check_data_statistics(self, task_id, support_loader, query_loader):
        """Task별 데이터셋 정보 출력"""
        print(f"\n[{'='*10} Data Statistics (Task {task_id}) {'='*10}]")
        print(f"  - Support Set (Train) Size: {len(support_loader.dataset)}")
        print(f"  - Query Set (Test) Size   : {len(query_loader.dataset)}")
        
        try:
            batch = next(iter(support_loader))
            # [수정] 딕셔너리 키 접근
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
                print(f"  - [Batch] Image Shape : {images.shape}")
                print(f"  - [Batch] Label Shape : {labels.shape}")
                if 'masks' in batch:
                    print(f"  - [Batch] Mask Shape  : {batch['masks'].shape}")
                else:
                    print(f"  - [Batch] Mask Shape  : None (Incremental Session)")
            else:
                print(f"  - [Batch] Data Type: {type(batch)} (Expected dict)")
        except Exception as e:
            print(f"  - [Warning] Failed to inspect batch: {e}")
        print("="*45 + "\n")

    def train_task(self, task_id):
        print(f"\n{'='*20} [Phase 2] Incremental Task {task_id} Start {'='*20}")
        
        # 데이터 로드
        support_loader, query_loader = self.inc_loader.get_incremental_loaders(task_id)
        
        # 데이터 통계 확인
        self.check_data_statistics(task_id, support_loader, query_loader)
        
        # 1. Adaptation (Fine-tuning)
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
        
        # 1. Backbone & Head Freeze
        self.model.backbone.eval()
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        if hasattr(self.model, 'head') and self.model.head is not None:
            self.model.head.eval()
            for param in self.model.head.parameters():
                param.requires_grad = False
                
        # 2. Projector Unfreeze
        if hasattr(self.model, 'projector'):
            self.model.projector.train()
            for param in self.model.projector.parameters():
                param.requires_grad = True
            
            # Optimizer 설정 (Projector 파라미터만)
            optimizer = optim.SGD(
                self.model.projector.parameters(),
                lr=self.args.lr_inc, 
                momentum=0.9
            )
        else:
            print("[Warning] No projector found in model. Skipping adaptation.")
            return
        
        iterator = iter(support_loader)
        
        for step in range(self.args.adaptation_steps):
            try:
                batch_data = next(iterator)
            except StopIteration:
                iterator = iter(support_loader)
                batch_data = next(iterator)
            
            # [수정] 딕셔너리 데이터 처리
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            if 'masks' in batch_data:
                concept_masks = batch_data['masks'].to(self.device).float()
            else:
                concept_masks = None
                
            optimizer.zero_grad()
            
            # Forward Pass
            model_outputs = self.model(images)
            targets = {'label': labels, 'masks': concept_masks}

            loss, log_dict = self.criterion(model_outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 10 == 0:
                print(f"    Step [{step+1}/{self.args.adaptation_steps}] Loss: {loss.item():.4f}")

        # 학습 종료 후 다시 Eval 모드
        self.model.eval()

    def register_prototypes(self, support_loader, task_id):
        """새로운 태스크의 데이터로 프로토타입 갱신/등록"""
        print(f"[*] Registering Prototypes for Task {task_id}...")
        self.model.eval()
        
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in support_loader:
                # [수정] 딕셔너리 데이터 처리
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                    
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
                # [수정] 딕셔너리 데이터 처리
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)

                # Forward Pass (logits는 프로토타입과의 거리 기반 점수)
                logits, _ = self.model(images)
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return acc