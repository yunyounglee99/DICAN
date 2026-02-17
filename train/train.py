"""
DICAN Training Pipeline - 3-Phase Base Training

Usage:
    python train.py \
      --data_path /root/DICAN_DATASETS/DDR \
      --epochs_base 30 \
      --batch_size 32 \
      --n_tasks 4 \
      --n_shot 10 \
      --device cuda
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from data.base_loader import DDRBaseDataset
from data.inc_loader import get_incremental_loader
from models.dican_cbm import DICAN_CBM
from train_base import BaseTrainer
from train_incremental import IncrementalTrainer
from utils.metrics import Evaluator


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="DICAN Training Pipeline")
    
    parser.add_argument('--dataset', type=str, default='DDR')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--n_concepts', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=5)
    
    parser.add_argument('--epochs_base', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_base', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_c', type=float, default=1.0)
    
    parser.add_argument('--n_tasks', type=int, default=4)
    parser.add_argument('--n_shot', type=int, default=10)
    parser.add_argument('--lr_inc', type=float, default=1e-3)
    parser.add_argument('--adaptation_steps', type=int, default=100)
    
    return parser.parse_args()


class IncLoaderManager:
    def __init__(self, args):
        self.args = args
        self.task_to_session = {1: 1, 2: 2, 3: 3}

    def get_incremental_loaders(self, task_id, mode_override=None):
        session_id = self.task_to_session.get(task_id)
        if session_id is None:
            return None, None

        if mode_override == 'test':
            query_loader = get_incremental_loader(
                session_id=session_id,
                data_root=self.args.data_path,
                mode='test',
                batch_size=self.args.batch_size,
                shot=None
            )
            return None, query_loader

        support_loader = get_incremental_loader(
            session_id=session_id,
            data_root=self.args.data_path,
            mode='train',
            batch_size=self.args.batch_size,
            shot=self.args.n_shot
        )
        
        query_loader = get_incremental_loader(
            session_id=session_id,
            data_root=self.args.data_path,
            mode='test',
            batch_size=self.args.batch_size,
            shot=None
        )
        
        return support_loader, query_loader


def get_base_loaders(args):
    print(f"[*] Loading Base Data (DDR) from {args.data_path}...")
    
    train_ds = DDRBaseDataset(root_dir=args.data_path, split='train')
    val_ds = DDRBaseDataset(root_dir=args.data_path, split='valid')
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.save_path, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"🚀 DICAN Training Start (3-Phase Pipeline)")
    print(f"   - Data Root: {args.data_path}")
    print(f"   - Device: {device}")
    print(f"   - Concepts: {args.n_concepts}")
    print(f"   - Hybrid Pooling: Max + Mean = {args.n_concepts * 2} dim")
    print("=" * 50 + "\n")

    # 모델 초기화
    model = DICAN_CBM(
        num_concepts=args.n_concepts,
        num_classes=args.num_classes,
        feature_dim=2048
    ).to(device)

    # Loader
    inc_loader_manager = IncLoaderManager(args)
    train_loader, val_loader = get_base_loaders(args)
    
    # Evaluator
    evaluator = Evaluator(model, device, val_loader, inc_loader_manager, args)

    # -------------------------------------------------------
    # [Phase 1] Base Training (3-Phase: Pretrain → Extract → Head)
    # -------------------------------------------------------
    base_trainer = BaseTrainer(args, model, device, train_loader, val_loader)
    model = base_trainer.run()
    
    # Base 평가
    print("\n[Eval] Evaluation after Base Session...")
    # eval 모드로 전환 (Projector는 base 모드이므로 identity)
    model.set_session_mode('eval')
    evaluator.evaluate_all_tasks(current_session_id=0)
    metrics = evaluator.calculate_metrics(current_session_id=0)
    print(f"   >>> Base Avg Acc: {metrics['avg_acc']:.2f}%")

    # -------------------------------------------------------
    # [Phase 2] Incremental Learning
    # -------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"🔄 Starting Incremental Phase ({args.n_tasks - 1} tasks)")
    print("=" * 50)
    
    model.set_session_mode('incremental')
    inc_trainer = IncrementalTrainer(args, model, device, inc_loader_manager)
    
    for task_id in range(1, args.n_tasks):
        acc = inc_trainer.train_task(task_id)
        
        # 평가
        model.set_session_mode('eval')
        evaluator.evaluate_all_tasks(current_session_id=task_id)
        metrics = evaluator.calculate_metrics(current_session_id=task_id)
        
        print(f"\n📊 [Metrics - Task {task_id}]")
        print(f"   - Average Accuracy  : {metrics['avg_acc']:.2f}%")
        print(f"   - Backward Transfer : {metrics['bwt']:.2f}%")
        print(f"   - Forward Transfer  : {metrics['fwt']:.2f}%")
        print(f"   - Forgetting        : {metrics['forgetting']:.2f}%")
        print(f"   - Task Accuracies   : {metrics['raw_accs']}")
        
        # 다음 태스크를 위해 incremental 모드 복귀
        model.set_session_mode('incremental')

    # 최종 결과
    print("\n" + "=" * 50)
    print("🎉 All Training Finished!")
    final = evaluator.calculate_metrics(current_session_id=args.n_tasks - 1)
    print(f"   - Final Avg Acc : {final['avg_acc']:.2f}%")
    print(f"   - Final BWT     : {final['bwt']:.2f}%")
    print(f"   - Final FWT     : {final['fwt']:.2f}%")
    print(f"   - Final Forget  : {final['forgetting']:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()