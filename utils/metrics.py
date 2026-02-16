import torch
import numpy as np
from copy import deepcopy

class Evaluator:
    def __init__(self, model, device, base_val_loader, inc_loader_manager, args):
        self.model = model
        self.device = device
        self.base_val_loader = base_val_loader
        self.inc_loader_manager = inc_loader_manager
        self.args = args
        
        # Performance Matrix R[i][j]: Session i가 끝난 후 Task j의 정확도
        # shape: (n_tasks, n_tasks)
        self.R = np.zeros((args.n_tasks, args.n_tasks))
        
        # FWT를 위한 초기 정확도 저장 (Random Init 상태 혹은 Base 학습 직후 상태)
        self.fwt_matrix = np.zeros((args.n_tasks, args.n_tasks))

    def evaluate_single_task(self, loader):
        """단일 태스크에 대한 평가 수행"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in loader:
                # 데이터 로더 형식에 따라 처리 (Dictionary vs Tuple)
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(self.device)
                    labels = batch_data['label'].to(self.device)
                else:
                    images, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                
                # 모델 Forward
                # DICAN 모델 특성상 output이 dict일 수 있음
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100.0 * correct / total
        return acc

    def evaluate_all_tasks(self, current_session_id):
        """
        현재 세션(current_session_id)까지 학습된 모델로
        0번부터 (n_tasks-1)번까지 모든 태스크를 평가하여 Matrix R에 기록
        """
        print(f"\n[*] Evaluating on All Tasks (Current Session: {current_session_id})...")
        
        for task_id in range(self.args.n_tasks):
            # 1. Loader 가져오기
            if task_id == 0:
                loader = self.base_val_loader
            else:
                # IncLoaderManager를 통해 Test Loader 가져오기 (shot=None -> Test set)
                _, loader = self.inc_loader_manager.get_incremental_loaders(task_id, mode_override='test')
            
            # 2. 평가 수행
            acc = self.evaluate_single_task(loader)
            self.R[current_session_id][task_id] = acc
            print(f"    -> Task {task_id} Acc: {acc:.2f}%")

    def calculate_metrics(self, current_session_id):
        """현재 세션 시점에서의 Metric 계산"""
        
        # 1. Average Accuracy (현재 세션까지 학습한 태스크들의 평균 정확도)
        avg_acc = np.mean(self.R[current_session_id, :current_session_id+1])
        
        # 2. Backward Transfer (BWT)
        # 공식: 1/(k) * sum(R[k][i] - R[i][i]) for i < k
        bwt = 0.0
        if current_session_id > 0:
            for i in range(current_session_id):
                bwt += (self.R[current_session_id][i] - self.R[i][i])
            bwt /= current_session_id
            
        # 3. Forgetting (망각)
        # 공식: max(previous_acc) - current_acc
        forgetting = 0.0
        if current_session_id > 0:
            for i in range(current_session_id):
                # 해당 태스크의 과거 최고 기록
                max_acc = np.max(self.R[:current_session_id, i])
                forgetting += (max_acc - self.R[current_session_id][i])
            forgetting /= current_session_id

        # 4. Forward Transfer (FWT)
        # 공식: 학습 전 성능 (Zero-shot) 측정. 보통 Random Init이나 Base 학습 직후 Future Task 성능을 봅니다.
        # 여기서는 R 행렬의 Upper Triangle 부분을 활용할 수도 있지만,
        # 편의상 R[i-1][i] (i번째 태스크 학습 직전의 i번째 태스크 성능)의 평균으로 계산합니다.
        fwt = 0.0
        if current_session_id > 0:
            # Task 1 ~ Current Task 까지
            for i in range(1, current_session_id + 1):
                # Task i를 학습하기 직전 Session(i-1)에서의 Task i 성능
                fwt += self.R[i-1][i] 
            fwt /= current_session_id

        return {
            "avg_acc": avg_acc,
            "bwt": bwt,
            "forgetting": forgetting,
            "fwt": fwt,
            "raw_accs": self.R[current_session_id, :current_session_id+1]
        }