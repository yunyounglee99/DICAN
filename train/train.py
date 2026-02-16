"""
Using Example :

python train.py \
  --data_path /Volumes/Nyoungs_SSD/macbook/dev/datasets/DICAN_DATASETS/DDR \
  --dataset DDR \
  --epochs_base 20 \
  --batch_size 32 \
  --n_tasks 4 \
  --n_shot 10 \
  --device cuda
  
"""

import argparse
import os
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
    parser.add_argument('--lr_base', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
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

    def get_incremental_loaders(self, task_id):
        session_id = self.task_to_session.get(task_id)
        if session_id is None:
            raise ValueError(f"[Error] No session mapped for Task ID {task_id}")

        print(f"[*] Loading Incremental Data for Task {task_id} (Session {session_id})...")
        
        # Support Set (Train, Few-shot)
        support_loader = get_incremental_loader(
            session_id=session_id,
            data_root=self.args.data_path,
            mode='train',
            batch_size=self.args.batch_size, # 메모리가 적다면 줄여야 함
            shot=self.args.n_shot
        )
        
        # Query Set (Test, All Data)
        query_loader = get_incremental_loader(
            session_id=session_id,
            data_root=self.args.data_path,
            mode='test', # 공정한 평가를 위해 test 셋 사용
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
    # 1-1. Base Loader 준비
    train_loader, val_loader = get_base_loaders(args)
    
    # 1-2. Mode 설정 (Backbone/Head 학습, Projector Freeze)
    model.set_session_mode('base')
    
    # 1-3. Trainer 실행
    base_trainer = BaseTrainer(args, model, device, train_loader, val_loader)
    
    # [추가 요청] 데이터 통계 확인
    if hasattr(base_trainer, 'check_data_statistics'):
        base_trainer.check_data_statistics()
        
    model = base_trainer.run() # 학습 및 Prototype 초기화 완료

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
    
    acc_history = []
    
    # Task Loop
    for task_id in range(1, args.n_tasks):
        # train_task 내부에서 check_data_statistics 호출됨 (수정된 코드 기준)
        acc = inc_trainer.train_task(task_id)
        acc_history.append(acc)
        
        print(f"📈 [Result] Task {task_id} Accuracy: {acc:.2f}%")
        print(f"    Current Avg Acc: {sum(acc_history)/len(acc_history):.2f}%")

    print("\n" + "="*50)
    print("🎉 All Training Finished!")
    print(f"   - Final Average Accuracy: {sum(acc_history)/len(acc_history):.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()