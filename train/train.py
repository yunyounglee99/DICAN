"""
DICAN Training Pipeline - DDR + FGADR Combined

Usage:
    python train.py \
      --data_path /root/DICAN_DATASETS \
      --epochs_base 30 \
      --batch_size 32 \
      --device cuda

[변경사항]
- data_path가 이제 DDR/FGADR 상위 폴더 (DICAN_DATASETS)를 가리킴
- DDR + FGADR을 ConcatDataset으로 합쳐서 Phase 1-A/B/C에 사용
- --fgadr_root 인자 추가 (별도 경로 지정 가능)
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
    
    # ★ data_path: DICAN_DATASETS 루트 (DDR, FGADR, aptos 등 상위 폴더)
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
    
    # 기본 경로 설정
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
    """
    ★ DDR + FGADR 합산 데이터 로더
    
    Before: DDR만 사용 → 6835 이미지, 275 마스크 (4%)
    After:  DDR + FGADR → ~8677 이미지, ~2117 마스크 (24%)
    
    두 데이터셋의 __getitem__ 반환 포맷이 동일하므로
    ConcatDataset으로 바로 합칠 수 있음.
    """
    print(f"\n[*] Loading Base Data...")
    print(f"    DDR:   {args.ddr_root}")
    print(f"    FGADR: {args.fgadr_root}")
    
    # ─── DDR ───
    ddr_train = DDRBaseDataset(root_dir=args.ddr_root, split='train')
    ddr_val = DDRBaseDataset(root_dir=args.ddr_root, split='valid')
    
    # ─── FGADR ───
    use_fgadr = (not args.no_fgadr) and os.path.exists(args.fgadr_root)
    
    if use_fgadr:
        fgadr_train = FGADRSegDataset(root_dir=args.fgadr_root, split='train')
        fgadr_val = FGADRSegDataset(root_dir=args.fgadr_root, split='valid')
        
        # ─── 합산 ───
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
        
        model.set_session_mode('eval')
        evaluator.evaluate_all_tasks(current_session_id=task_id)
        metrics = evaluator.calculate_metrics(current_session_id=task_id)
        
        print(f"\n📊 [Metrics - Task {task_id}]")
        print(f"   - Average Accuracy  : {metrics['avg_acc']:.2f}%")
        print(f"   - Backward Transfer : {metrics['bwt']:.2f}%")
        print(f"   - Forward Transfer  : {metrics['fwt']:.2f}%")
        print(f"   - Forgetting        : {metrics['forgetting']:.2f}%")
        print(f"   - Task Accuracies   : {metrics['raw_accs']}")
        
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