import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# 프로젝트 모듈 임포트
from utils.loss import TaskLoss, ConceptLoss
from models.prototypes import PrototypeManager

class IncrementalTrainer:
    """
    DICAN 모델의 Incremental Task (Phase 2)를 담당하는 트레이너 클래스입니다.
    새로운 Task(Domain/Class)가 들어올 때마다:
    1. Adaptation: Few-shot 데이터를 이용해 모델을 미세조정 (Optional)
    2. Registration: 새로운 데이터의 Concept Feature를 추출하여 프로토타입 등록
    3. Evaluation: 업데이트된 지식으로 성능 평가
    를 수행합니다.
    """
    def __init__(self, args, model, device, inc_loader):
        self.args = args
        self.model = model
        self.device = device
        self.inc_loader = inc_loader # IncLoader 인스턴스
        
        # --------------------------------------------------------
        # 1. Prototype Manager 초기화
        # --------------------------------------------------------
        # Base Training에서 생성된 프로토타입 상태를 이어받습니다.
        self.prototype_manager = PrototypeManager(self.args, self.device)
        
        # 모델에 이미 저장된 프로토타입이 있다면 매니저에 동기화
        if hasattr(self.model, 'prototypes') and self.model.prototypes is not None:
            self.prototype_manager.prototypes = self.model.prototypes.clone()
            print(f"[*] Loaded existing prototypes from model: Shape {self.prototype_manager.prototypes.shape}")

        # --------------------------------------------------------
        # 2. Loss Function 설정 (Adaptation용)
        # --------------------------------------------------------
        self.criterion_task = TaskLoss()
        self.criterion_concept = ConceptLoss()
        self.lambda_c = getattr(self.args, 'lambda_c', 1.0) # Concept Loss 가중치

    def train_task(self, task_id):
        """특정 Incremental Task에 대한 전체 학습/평가 파이프라인"""
        print(f"\n{'='*20} [Phase 2] Incremental Task {task_id} Start {'='*20}")
        
        # 1. 데이터 로드 (Support Set: 학습/등록용, Query Set: 평가용)
        # inc_loader.get_task_loaders(task_id)가 (support_loader, query_loader)를 반환한다고 가정
        support_loader, query_loader = self.inc_loader.get_task_loaders(task_id)
        
        # 2. Adaptation (Fine-tuning)
        # Few-shot 데이터로 도메인 적응 수행. adaptation_steps가 0이면 수행 안 함.
        if self.args.adaptation_steps > 0:
            self.adapt_to_domain(support_loader, task_id)
            
        # 3. Prototype Registration (핵심)
        # 적응된 모델을 이용해 새로운 데이터들의 'Concept 중심점(Prototype)'을 계산하고 등록
        self.register_prototypes(support_loader, task_id)
        
        # 4. Evaluation
        # 업데이트된 프로토타입을 기반으로 현재 Task 평가
        acc = self.evaluate(query_loader, task_id)
        
        print(f"{'='*20} Incremental Task {task_id} Finished (Acc: {acc:.2f}%) {'='*20}\n")
        return acc

    def adapt_to_domain(self, support_loader, task_id):
        """
        Support Set을 사용하여 모델을 현재 도메인에 맞게 미세조정(Fine-tuning)합니다.
        Incremental Learning에서는 망각(Forgetting)을 방지하기 위해 
        Learning Rate를 낮게 설정하거나 특정 Layer만 학습하는 것이 일반적입니다.
        """
        print(f"[*] Adapting to Task {task_id} (Fine-tuning {self.args.adaptation_steps} steps)...")
        
        self.model.train()
        
        # Optimizer 설정: Base보다 낮은 LR 사용 (args.lr_inc)
        # SGD가 Few-shot에서 일반화 성능이 더 좋을 때가 많음
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.args.lr_inc, 
            momentum=0.9,
            weight_decay=self.args.weight_decay
        )
        
        # 데이터 로더를 이터레이터로 변환 (Step 단위 학습을 위해)
        iterator = iter(support_loader)
        
        for step in range(self.args.adaptation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(support_loader)
                batch = next(iterator)
            
            # 데이터 언패킹 (Mask 유무 확인)
            if len(batch) == 3:
                images, labels, concept_masks = batch
                concept_masks = concept_masks.to(self.device).float()
            else:
                images, labels = batch
                concept_masks = None
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            logits, concepts = self.model(images)
            
            # Loss Calculation
            # 1. Task Loss (필수)
            loss = self.criterion_task(logits, labels)
            
            # 2. Concept Loss (Mask가 제공되는 경우에만)
            # Incremental 단계에서는 Mask가 없을 수도 있으므로 체크 필요
            if concept_masks is not None:
                loss_c = self.criterion_concept(concepts, concept_masks)
                loss += (self.lambda_c * loss_c)
                
            loss.backward()
            optimizer.step()
            
            # 로그 출력
            if (step + 1) % 10 == 0 or (step + 1) == self.args.adaptation_steps:
                print(f"    Step [{step+1}/{self.args.adaptation_steps}] Loss: {loss.item():.4f}")

    def register_prototypes(self, support_loader, task_id):
        """
        [DICAN 연구 핵심]
        적응된 모델(Feature Extractor)을 통과한 Feature들의 평균을 구해
        PrototypeManager에 등록합니다. 이를 통해 Classifier Head 없이도
        새로운 클래스/도메인을 분류할 수 있게 됩니다.
        """
        print(f"[*] Registering/Updating Prototypes for Task {task_id} using Support Set...")
        self.model.eval()
        
        all_concepts = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(support_loader, desc="Extracting Prototypes"):
                # Feature 추출에는 Mask가 필요 없음
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                # 모델의 bottleneck feature (concepts) 추출
                _, concepts = self.model(images)
                
                all_concepts.append(concepts.cpu())
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_concepts, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # PrototypeManager 업데이트 (새로운 클래스 추가 or 기존 클래스 갱신)
        self.prototype_manager.update_prototypes(features, labels, task_id)
        
        # 모델의 내부 프로토타입 버퍼 갱신 (Inference시 사용됨)
        # DICAN_CBM 모델이 'prototypes' 속성을 가지고 이를 이용해 분류한다고 가정
        self.model.prototypes = self.prototype_manager.get_prototypes()
        print(f"[*] Prototypes Updated. Current Classes: {len(self.prototype_manager.class_to_idx)}")

    def evaluate(self, query_loader, task_id):
        """
        Query Set(Test Data)을 이용한 성능 평가.
        모델은 업데이트된 프로토타입과 입력 이미지의 Concept Feature 간의
        유사도(Distance)를 기반으로 예측을 수행합니다.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(query_loader, desc=f"Evaluating Task {task_id}"):
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                # Forward Pass
                # 모델이 내부적으로 self.prototypes를 이용해 logits(유사도 점수)를 반환해야 함
                logits, _ = self.model(images)
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        return acc