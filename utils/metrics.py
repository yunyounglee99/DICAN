import torch
import numpy as np
from copy import deepcopy
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Quadratic Weighted Kappa (QWK) 계산.
    
    DR Grading의 표준 평가 지표로, 순서형(Ordinal) 분류에서
    단순 정확도보다 임상적으로 더 의미 있는 지표.
    
    - 1.0  : 완벽한 일치
    - 0.0  : 우연 수준의 일치
    - <0   : 우연보다 못한 일치
    
    일반적으로:
      0.81~1.00 = Almost Perfect
      0.61~0.80 = Substantial
      0.41~0.60 = Moderate
      0.21~0.40 = Fair
    """
    if len(y_true) == 0:
        return 0.0
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


class Evaluator:
    def __init__(self, model, device, base_val_loader, inc_loader_manager, args):
        self.model = model
        self.device = device
        self.base_val_loader = base_val_loader
        self.inc_loader_manager = inc_loader_manager
        self.args = args
        
        # Performance Matrix R[i][j]: Session i가 끝난 후 Task j의 정확도
        self.R = np.zeros((args.n_tasks, args.n_tasks))
        
        # ★ QWK Matrix: R과 동일 구조
        self.R_kappa = np.zeros((args.n_tasks, args.n_tasks))
        
        self.fwt_matrix = np.zeros((args.n_tasks, args.n_tasks))

    def evaluate_single_task(self, loader):
        """단일 태스크 평가 (Accuracy + QWK)"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in loader:
                if isinstance(batch_data, dict):
                    images = batch_data['image'].to(self.device)
                    labels = batch_data['label'].to(self.device)
                else:
                    images, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # ★ QWK용 수집
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = 100.0 * correct / total
        kappa = quadratic_weighted_kappa(all_labels, all_preds)
        return acc, kappa

    def evaluate_all_tasks(self, current_session_id):
        """모든 태스크 평가 → R matrix와 R_kappa matrix 모두 업데이트"""
        print(f"\n[*] Evaluating on All Tasks (Current Session: {current_session_id})...")
        
        for task_id in range(self.args.n_tasks):
            if task_id == 0:
                # self.model.set_eval_mode(task_id=0)
                loader = self.base_val_loader
            else:
                _, loader = self.inc_loader_manager.get_incremental_loaders(task_id, mode_override='test')
            
            acc, kappa = self.evaluate_single_task(loader)
            self.R[current_session_id][task_id] = acc
            self.R_kappa[current_session_id][task_id] = kappa
            print(f"    -> Task {task_id} Acc: {acc:.2f}%  |  QWK: {kappa:.4f}")

    def calculate_metrics(self, current_session_id):
        """현재 세션 시점의 종합 Metric (QWK 포함)"""
        
        # 1. Average Accuracy & QWK
        avg_acc = np.mean(self.R[current_session_id, :current_session_id+1])
        avg_kappa = np.mean(self.R_kappa[current_session_id, :current_session_id+1])
        
        # 2. Backward Transfer (BWT)
        bwt = 0.0
        bwt_kappa = 0.0
        if current_session_id > 0:
            for i in range(current_session_id):
                bwt += (self.R[current_session_id][i] - self.R[i][i])
                bwt_kappa += (self.R_kappa[current_session_id][i] - self.R_kappa[i][i])
            bwt /= current_session_id
            bwt_kappa /= current_session_id
            
        # 3. Forgetting
        forgetting = 0.0
        forgetting_kappa = 0.0
        if current_session_id > 0:
            for i in range(current_session_id):
                max_acc = np.max(self.R[:current_session_id, i])
                forgetting += (max_acc - self.R[current_session_id][i])
                max_kappa = np.max(self.R_kappa[:current_session_id, i])
                forgetting_kappa += (max_kappa - self.R_kappa[current_session_id][i])
            forgetting /= current_session_id
            forgetting_kappa /= current_session_id

        # 4. Forward Transfer (FWT)
        fwt = 0.0
        fwt_kappa = 0.0
        if current_session_id > 0:
            for i in range(1, current_session_id + 1):
                fwt += self.R[i-1][i]
                fwt_kappa += self.R_kappa[i-1][i]
            fwt /= current_session_id
            fwt_kappa /= current_session_id

        return {
            "avg_acc": avg_acc,
            "avg_kappa": avg_kappa,
            "bwt": bwt,
            "bwt_kappa": bwt_kappa,
            "forgetting": forgetting,
            "forgetting_kappa": forgetting_kappa,
            "fwt": fwt,
            "fwt_kappa": fwt_kappa,
            "raw_accs": self.R[current_session_id, :current_session_id+1],
            "raw_kappas": self.R_kappa[current_session_id, :current_session_id+1],
        }