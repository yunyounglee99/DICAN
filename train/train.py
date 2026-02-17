"""
DICAN Training Pipeline - DDR + FGADR Combined

Usage:
    python train.py \
      --data_path /root/DICAN_DATASETS \
      --epochs_base 30 \
      --batch_size 32 \
      --device cuda

★ QWK (Quadratic Weighted Kappa) 추가: 모든 평가 지점에서 출력
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset

from data.base_loader import DDRBaseDataset
from data.fgadr_loader import FGADRSegDataset
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
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root of all datasets (e.g., /root/DICAN_DATASETS)')
    parser.add_argument('--ddr_root', type=str, default=None,
                        help='DDR dataset root. Default: {data_path}/DDR')
    parser.add_argument('--fgadr_root', type=str, default=None,
                        help='FGADR dataset root. Default: {data_path}/FGADR')
    parser.add_argument('--no_fgadr', action='store_true',
                        help='Disable FGADR (use DDR only)')
    
    parser.add_argument('--dataset', type=str, default='DDR')
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
    
    args = parser.parse_args()
    
    if args.ddr_root is None:
        args.ddr_root = os.path.join(args.data_path, 'DDR')
    if args.fgadr_root is None:
        args.fgadr_root = os.path.join(args.data_path, 'FGADR')
    
    return args


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
    print(f"\n[*] Loading Base Data...")
    print(f"    DDR:   {args.ddr_root}")
    print(f"    FGADR: {args.fgadr_root}")
    
    ddr_train = DDRBaseDataset(root_dir=args.ddr_root, split='train')
    ddr_val = DDRBaseDataset(root_dir=args.ddr_root, split='valid')
    
    use_fgadr = (not args.no_fgadr) and os.path.exists(args.fgadr_root)
    
    if use_fgadr:
        fgadr_train = FGADRSegDataset(root_dir=args.fgadr_root, split='train')
        fgadr_val = FGADRSegDataset(root_dir=args.fgadr_root, split='valid')
        
        train_dataset = ConcatDataset([ddr_train, fgadr_train])
        val_dataset = ConcatDataset([ddr_val, fgadr_val])
        
        print(f"\n    ✅ Combined Dataset:")
        print(f"       Train: DDR({len(ddr_train)}) + FGADR({len(fgadr_train)}) = {len(train_dataset)}")
        print(f"       Valid: DDR({len(ddr_val)}) + FGADR({len(fgadr_val)}) = {len(val_dataset)}")
    else:
        if not os.path.exists(args.fgadr_root):
            print(f"    ⚠️ FGADR not found at {args.fgadr_root}, using DDR only")
        else:
            print(f"    ℹ️ FGADR disabled (--no_fgadr)")
        
        train_dataset = ddr_train
        val_dataset = ddr_val
        
        print(f"\n    DDR Only:")
        print(f"       Train: {len(train_dataset)}")
        print(f"       Valid: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.save_path, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"🚀 DICAN Training Start (DDR + FGADR)")
    print(f"   - Data Root: {args.data_path}")
    print(f"   - DDR:       {args.ddr_root}")
    print(f"   - FGADR:     {args.fgadr_root}")
    print(f"   - Device: {device}")
    print(f"   - Concepts: {args.n_concepts}")
    print(f"   - Hybrid Pooling: Max + Mean = {args.n_concepts * 2} dim")
    print("=" * 50 + "\n")

    model = DICAN_CBM(
        num_concepts=args.n_concepts,
        num_classes=args.num_classes,
        feature_dim=2048
    ).to(device)

    inc_loader_manager = IncLoaderManager(args)
    train_loader, val_loader = get_base_loaders(args)
    
    evaluator = Evaluator(model, device, val_loader, inc_loader_manager, args)

    # -------------------------------------------------------
    # [Phase 1] Base Training
    # -------------------------------------------------------
    base_trainer = BaseTrainer(args, model, device, train_loader, val_loader)
    model = base_trainer.run()
    
    # ★ Base 평가 (QWK 포함)
    print("\n[Eval] Evaluation after Base Session...")
    model.set_session_mode('eval')
    evaluator.evaluate_all_tasks(current_session_id=0)
    metrics = evaluator.calculate_metrics(current_session_id=0)
    print(f"   >>> Base Avg Acc: {metrics['avg_acc']:.2f}%, Avg QWK: {metrics['avg_kappa']:.4f}")

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
        
        model.set_session_mode('eval')
        evaluator.evaluate_all_tasks(current_session_id=task_id)
        metrics = evaluator.calculate_metrics(current_session_id=task_id)
        
        # ★ QWK 포함 전체 Metric 출력
        print(f"\n📊 [Metrics - Task {task_id}]")
        print(f"   - Average Accuracy  : {metrics['avg_acc']:.2f}%")
        print(f"   - Average QWK       : {metrics['avg_kappa']:.4f}")
        print(f"   - Backward Transfer : {metrics['bwt']:.2f}%  (QWK: {metrics['bwt_kappa']:.4f})")
        print(f"   - Forward Transfer  : {metrics['fwt']:.2f}%  (QWK: {metrics['fwt_kappa']:.4f})")
        print(f"   - Forgetting        : {metrics['forgetting']:.2f}%  (QWK: {metrics['forgetting_kappa']:.4f})")
        print(f"   - Task Accuracies   : {metrics['raw_accs']}")
        print(f"   - Task QWKs         : {metrics['raw_kappas']}")
        
        model.set_session_mode('incremental')

    # ★ 최종 결과 (QWK 포함)
    print("\n" + "=" * 50)
    print("🎉 All Training Finished!")
    final = evaluator.calculate_metrics(current_session_id=args.n_tasks - 1)
    print(f"   - Final Avg Acc  : {final['avg_acc']:.2f}%")
    print(f"   - Final Avg QWK  : {final['avg_kappa']:.4f}")
    print(f"   - Final BWT      : {final['bwt']:.2f}%  (QWK: {final['bwt_kappa']:.4f})")
    print(f"   - Final FWT      : {final['fwt']:.2f}%  (QWK: {final['fwt_kappa']:.4f})")
    print(f"   - Final Forget   : {final['forgetting']:.2f}%  (QWK: {final['forgetting_kappa']:.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()