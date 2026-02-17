import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils.loss import DICANLoss


class IncrementalTrainer:
    """
    DICAN Incremental Trainer - 3-Phase 구조 대응
    
    Phase 2에서 호출됨:
    - Backbone, Head, Prototype 모두 Frozen
    - Projector만 학습하여 새 도메인 적응
    """
    def __init__(self, args, model, device, inc_loader):
        self.args = args
        self.model = model
        self.device = device
        self.inc_loader = inc_loader
        
        self.criterion = DICANLoss(
            mode='incremental',
            num_concepts=self.args.n_concepts,
            num_classes=self.args.num_classes
        ).to(self.device)

    def train_task(self, task_id):
        print(f"\n{'='*20} [Phase 2] Incremental Task {task_id} {'='*20}")
        
        support_loader, query_loader = self.inc_loader.get_incremental_loaders(task_id)
        
        if support_loader is None:
            print(f"[Warning] No data for task {task_id}")
            return 0.0
        
        self._print_stats(task_id, support_loader, query_loader)
        
        # 1. Adaptation
        if self.args.adaptation_steps > 0:
            self.adapt_to_domain(support_loader, task_id)
        
        # 2. Evaluation
        acc = self.evaluate(query_loader, task_id)
        
        print(f"{'='*20} Task {task_id} Done (Acc: {acc:.2f}%) {'='*20}\n")
        return acc

    def adapt_to_domain(self, support_loader, task_id):
        """Projector만 미세조정"""
        print(f"[*] Adapting Projector for Task {task_id}...")
        
        self.model.set_session_mode('incremental')
        
        # Projector 파라미터만 학습
        trainable_params = self.model.get_trainable_params()
        if not trainable_params:
            print("[Warning] No trainable parameters found!")
            return
        
        optimizer = optim.SGD(
            trainable_params,
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
            
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(images)
            targets = {'label': labels, 'masks': None}
            
            loss, log_dict = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 20 == 0:
                print(f"    Step [{step+1}/{self.args.adaptation_steps}] "
                      f"Loss: {loss.item():.4f} "
                      f"(align={log_dict['loss_align']:.3f}, "
                      f"ord={log_dict['loss_ordinal']:.3f})")

    def evaluate(self, query_loader, task_id):
        """Query Set 평가"""
        self.model.set_session_mode('eval')
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in tqdm(query_loader, desc=f"Eval Task {task_id}"):
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                outputs = self.model(images)
                logits = outputs['logits']
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total if total > 0 else 0.0

    def _print_stats(self, task_id, support_loader, query_loader):
        print(f"\n  Task {task_id} Data:")
        print(f"    Support: {len(support_loader.dataset)} samples")
        print(f"    Query:   {len(query_loader.dataset)} samples")