"""
Using Example :

python train.py \
  --data_path /root/DICAN_DATASETS/DDR \
  --dataset DDR \
  --epochs_base 30 \
  --batch_size 32 \
  --n_tasks 4 \
  --n_shot 10 \
  --device cuda > dican_ex.txt 2>&1

"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 1. 사용자 정의 모듈 Import (파일 경로 기반)
# -----------------------------------------------------------------------------
from data.base_loader import DDRBaseDataset
from data.inc_loader import get_incremental_loader
from models.dican_cbm import DICAN_CBM
from train_base import BaseTrainer
from train_incremental import IncrementalTrainer
from utils.metrics import Evaluator

# -----------------------------------------------------------------------------
# 2. 유틸리티 함수
# -----------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description="DICAN Training Pipeline")
    
    # [Data]
    parser.add_argument('--dataset', type=str, default='DDR', help='Base dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Root path of DICAN_DATASETS')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Save path')
    
    # [System]
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    # [Model]
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone type')
    parser.add_argument('--n_concepts', type=int, default=4, help='Number of concepts (bottleneck)')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of DR grades')
    
    # [Base Training]
    parser.add_argument('--epochs_base', type=int, default=20, help='Base training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Base batch size')
    parser.add_argument('--lr_base', type=float, default=1e-2, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--lambda_c', type=float, default=1.0, help='Concept loss weight')
    
    # [Incremental Training]
    parser.add_argument('--n_tasks', type=int, default=4, help='Total tasks (Base + 3 Inc)')
    parser.add_argument('--n_shot', type=int, default=10, help='Few-shot count')
    parser.add_argument('--lr_inc', type=float, default=1e-3, help='Incremental learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=100, help='Adaptation steps')
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 3. 데이터 로더 래퍼 (Loader Wrapper)
# -----------------------------------------------------------------------------
class IncLoaderManager:
    """
    IncrementalTrainer가 요구하는 인터페이스(get_incremental_loaders)를
    data/inc_loader.py의 함수와 연결해주는 래퍼 클래스
    """
    def __init__(self, args):
        self.args = args
        # Task ID와 Session ID 매핑 (Task 1 -> Session 1: APTOS, etc.)
        self.task_to_session = {
            1: 1, # APTOS
            2: 2, # Messidor-2
            3: 3  # DRAC22
        }

    def get_incremental_loaders(self, task_id, mode_override=None):
        session_id = self.task_to_session.get(task_id)
        if session_id is None:
            # Task ID 0은 Base Loader이므로 None 리턴 혹은 에러 처리
            return None, None

        if mode_override == 'test':
            # Test Loader만 필요할 때
            print(f"[*] (Eval) Loading Test Data for Task {task_id}...")
            query_loader = get_incremental_loader(
                session_id=session_id,
                data_root=self.args.data_path,
                mode='test',
                batch_size=self.args.batch_size,
                shot=None
            )
            return None, query_loader

        print(f"[*] Loading Incremental Data for Task {task_id} (Session {session_id})...")
        
        # Support Set (Train, Few-shot)
        support_loader = get_incremental_loader(
            session_id=session_id,
            data_root=self.args.data_path,
            mode='train',
            batch_size=self.args.batch_size,
            shot=self.args.n_shot
        )
        
        # Query Set (Test, All Data)
        query_loader = get_incremental_loader(
            session_id=session_id,
            data_root=self.args.data_path,
            mode='test',
            batch_size=self.args.batch_size,
            shot=None
        )
        
        return support_loader, query_loader

def get_base_loaders(args):
    """base_loader.py의 DDRBaseDataset을 DataLoader로 포장"""
    print(f"[*] Loading Base Data (DDR) from {args.data_path}...")
    
    train_ds = DDRBaseDataset(root_dir=args.data_path, split='train')
    val_ds = DDRBaseDataset(root_dir=args.data_path, split='valid')
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# -----------------------------------------------------------------------------
# 4. Main Function
# -----------------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("\n" + "="*50)
    print(f"🚀 DICAN Training Start")
    print(f"   - Data Root: {args.data_path}")
    print(f"   - Device: {device}")
    print(f"   - Concepts: {args.n_concepts}")
    print("="*50 + "\n")

    # 1. 모델 초기화
    model = DICAN_CBM(
        num_concepts=args.n_concepts,
        num_classes=args.num_classes,
        feature_dim=2048 # ResNet50 기준
    ).to(device)

    # -------------------------------------------------------
    # [Phase 1] Base Training (Task 0)
    # -------------------------------------------------------
    
    # Loader Manager 초기화 (Eval 용)
    inc_loader_manager = IncLoaderManager(args)
    
    # 1-1. Base Loader 준비
    train_loader, val_loader = get_base_loaders(args)
    
    # [NEW] Evaluator 초기화
    # Base Val Loader는 Task 0 평가용으로 전달
    evaluator = Evaluator(model, device, val_loader, inc_loader_manager, args)

    model.set_session_mode('base')
    base_trainer = BaseTrainer(args, model, device, train_loader, val_loader)
    
    if hasattr(base_trainer, 'check_data_statistics'):
        base_trainer.check_data_statistics()
        
    model = base_trainer.run() 
    
    # [EVAL] Base Session 종료 후 평가 (Task 0 완료 시점)
    # 이때 Future Task (Task 1, 2, 3...)에 대해서도 평가를 수행해야 FWT를 잴 수 있음
    print("\n[Eval] Evaluation after Base Session (Task 0)...")
    evaluator.evaluate_all_tasks(current_session_id=0)
    metrics = evaluator.calculate_metrics(current_session_id=0)
    print(f"   >>> Base Avg Acc: {metrics['avg_acc']:.2f}%")

    # -------------------------------------------------------
    # [Phase 2] Incremental Learning (Task 1 ~ N)
    # -------------------------------------------------------
    print("\n" + "="*50)
    print(f"🔄 Starting Incremental Phase (Total {args.n_tasks-1} tasks)")
    print("="*50)
    
    # 2-1. Inc Loader 준비
    inc_loader_manager = IncLoaderManager(args)
    
    # 2-2. Mode 설정 (Projector 학습, Backbone/Head Freeze)
    model.set_session_mode('incremental')
    
    # 2-3. Trainer 초기화 (PrototypeBank 연동)
    inc_trainer = IncrementalTrainer(args, model, device, inc_loader_manager)
    
    for task_id in range(1, args.n_tasks):
        # 1. 학습
        acc = inc_trainer.train_task(task_id)
        
        # 2. [EVAL] 전체 태스크 평가 및 Metric 계산
        # 현재 Task 학습이 끝났으므로, 과거~현재~미래 모든 태스크 성능 측정
        print(f"\n[Eval] Evaluation after Task {task_id}...")
        evaluator.evaluate_all_tasks(current_session_id=task_id)
        
        # 3. Metric 계산
        metrics = evaluator.calculate_metrics(current_session_id=task_id)
        
        print(f"\n📊 [Metrics - Task {task_id}]")
        print(f"   - Average Accuracy  : {metrics['avg_acc']:.2f}%")
        print(f"   - Backward Transfer : {metrics['bwt']:.2f}%  (High is good)")
        print(f"   - Forward Transfer  : {metrics['fwt']:.2f}%  (High is good)")
        print(f"   - Forgetting        : {metrics['forgetting']:.2f}% (Low is good)")
        print(f"   - Task Accuracies   : {metrics['raw_accs']}")
        print("-" * 30)

    print("\n" + "="*50)
    print("🎉 All Training Finished!")
    
    # 최종 결과 재출력
    final_metrics = evaluator.calculate_metrics(current_session_id=args.n_tasks-1)
    print(f"   - Final Average Accuracy : {final_metrics['avg_acc']:.2f}%")
    print(f"   - Final BWT              : {final_metrics['bwt']:.2f}%")
    print(f"   - Final FWT              : {final_metrics['fwt']:.2f}%")
    print(f"   - Final Forgetting       : {final_metrics['forgetting']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()