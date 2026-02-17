import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy
# Need F for mse_loss
import torch.nn.functional as F


from utils.loss import DICANLoss


class IncrementalTrainer:
    """
    DICAN Incremental Trainer - 강화된 Forgetting 방지 버전
    
    [핵심 변경사항 - 문제4 (Catastrophic Forgetting) 해결]
    
    1. Projector 초기화 재설정: 매 Task마다 identity에 가까운 상태로 리셋
       → 이전 Task에서 과도하게 변형된 Projector가 다음 Task에 악영향 방지
    
    2. L2 Anchor Regularization:
       adaptation 시작 시점의 Projector 가중치를 anchor로 저장하고,
       adaptation 중 anchor에서 너무 멀어지지 않도록 L2 페널티
       → Projector가 현재 Task에 과적합되는 것을 방지
    
    3. 학습률 Warmup:
       처음 20% step은 낮은 LR로 시작 → 점진적으로 올림
       → 급격한 가중치 변화 방지
    
    4. Concept Score Monitoring:
       adaptation 전후 concept score 변화를 출력하여
       Projector가 prototype space를 얼마나 왜곡하는지 추적
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
        
        # ★ Projector anchor (L2 reg 기준점)
        self.projector_anchor = None

    def train_task(self, task_id):
        print(f"\n{'='*20} [Phase 2] Incremental Task {task_id} {'='*20}")
        
        support_loader, query_loader = self.inc_loader.get_incremental_loaders(task_id)
        
        if support_loader is None:
            print(f"[Warning] No data for task {task_id}")
            return 0.0
        
        self._print_stats(task_id, support_loader, query_loader)
        
        # ★ Projector를 identity 근처로 re-initialize
        self._reset_projector_near_identity()
        
        # Adaptation
        if self.args.adaptation_steps > 0:
            self.adapt_to_domain(support_loader, task_id)
        
        # Evaluation
        acc = self.evaluate(query_loader, task_id)
        
        print(f"{'='*20} Task {task_id} Done (Acc: {acc:.2f}%) {'='*20}\n")
        return acc

    def _reset_projector_near_identity(self):
        """
        Projector를 identity mapping에 가깝게 재초기화.
        
        기존 문제: Task 1에서 학습된 projector가 Task 2 시작점이 됨
        → 누적된 왜곡이 catastrophic forgetting 유발
        
        해결: 매 Task 시작 시 "거의 identity"로 리셋
        → 각 Task는 clean state에서 최소한의 적응만 수행
        """
        print("    [*] Resetting Projector near identity...")
        for m in self.model.projector.modules():
            if isinstance(m, torch.nn.Conv2d):
                # Identity-like init for 1x1 conv
                # (2048, 2048, 1, 1)인 경우 대각선을 1로, 나머지는 작은 noise
                with torch.no_grad():
                    fan_in = m.weight.size(1)
                    fan_out = m.weight.size(0)
                    if fan_in == fan_out:
                        # Identity + small noise
                        identity = torch.eye(fan_in, device=m.weight.device).unsqueeze(-1).unsqueeze(-1)
                        noise = torch.randn_like(m.weight) * 0.01
                        m.weight.copy_(identity + noise)
                    else:
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def adapt_to_domain(self, support_loader, task_id):
        """Projector adaptation with L2 anchor regularization"""
        print(f"[*] Adapting Projector for Task {task_id}...")
        
        self.model.set_session_mode('incremental')
        
        trainable_params = self.model.get_trainable_params()
        if not trainable_params:
            print("[Warning] No trainable parameters found!")
            return
        
        # ★ Anchor 저장 (adaptation 시작 시점의 가중치)
        self.projector_anchor = {
            name: param.clone().detach() 
            for name, param in self.model.projector.named_parameters()
        }
        
        # ★ 학습률 및 스텝 조정
        base_lr = self.args.lr_inc * 0.5  # 기존보다 절반
        warmup_steps = max(self.args.adaptation_steps // 5, 10)
        
        optimizer = optim.SGD(
            trainable_params,
            lr=base_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # ★ L2 anchor 정규화 강도
        lambda_anchor = 0.1
        
        iterator = iter(support_loader)
        
        for step in range(self.args.adaptation_steps):
            try:
                batch_data = next(iterator)
            except StopIteration:
                iterator = iter(support_loader)
                batch_data = next(iterator)
            
            # ★ Warmup LR
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            
            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(images)
            targets = {'label': labels, 'masks': None}
            
            loss_main, log_dict = self.criterion(outputs, targets)
            
            # ★ L2 Anchor Regularization
            loss_anchor = 0.0
            for name, param in self.model.projector.named_parameters():
                if name in self.projector_anchor:
                    loss_anchor += F.mse_loss(param, self.projector_anchor[name])
            
            loss = loss_main + lambda_anchor * loss_anchor
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            
            if (step + 1) % 20 == 0:
                print(f"    Step [{step+1}/{self.args.adaptation_steps}] "
                      f"Loss: {loss.item():.4f} "
                      f"(align={log_dict['loss_align']:.3f}, "
                      f"ord={log_dict['loss_ordinal']:.3f}, "
                      f"anchor={lambda_anchor*loss_anchor:.3f})")

    def evaluate(self, query_loader, task_id):
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


