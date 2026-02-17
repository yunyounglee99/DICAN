"""
DICAN DR Grading Dataset for Mammoth Framework
================================================
[변경사항]
- Task 0 (Base Session) 데이터 분리:
  ★ Phase 1-A: FGADR만 (100% 마스크) → backbone + seg 학습
  ★ Phase 1-B/C: DDR+FGADR 전체 → prototype 추출 + classification
- QWK (Quadratic Weighted Kappa) 메트릭 추가
- FGADR loader 연동 (fgadr_loader.py)

[Mammoth 비교실험용]
EWC, LwF, L2P, DualPrompt 등 다른 CL 모델은 seg 없이 classification만 하므로
Task 0에서 DDR+FGADR 전체(full)를 사용.
DICAN만 Phase 1-A에서 seg_loader를 별도로 쓰는 구조.
"""

import sys
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

# Mammoth 필수 유틸리티
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders

# -----------------------------------------------------------------------------
# [경로 설정]
# -----------------------------------------------------------------------------
PROJECT_ROOT = '/root/DICAN'
DATA_ROOT_DDR = '/root/DICAN_DATASETS/DDR'
DATA_ROOT_FGADR = '/root/DICAN_DATASETS/FGADR'
DATA_ROOT_INC = '/root/DICAN_DATASETS'

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from data.base_loader import DDRBaseDataset
    from data.fgadr_loader import FGADRSegDataset
    from data.inc_loader import UnifiedIncrementalDataset
except ImportError as e:
    print(f"[Error] DICAN 데이터 로더를 찾을 수 없습니다: {PROJECT_ROOT}")
    raise e


# =============================================================================
# Mammoth 호환 래퍼
# =============================================================================
class MammothWrapper(Dataset):
    """
    Standard CL 모델(EWC, LwF 등) 호환 래퍼.
    반환: (image, label, original_image) — Mammoth 필수 3-tuple.
    마스크는 제거 (standard 모델은 3채널 RGB만 입력).
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
        if hasattr(dataset, 'data_map'):       # DDR
            src = dataset.data_map
            self.data = np.array([d['img_name'] for d in src])
            self.targets = np.array([d['label'] for d in src])
        elif hasattr(dataset, 'data_list'):    # Inc (APTOS, Messidor, DRAC)
            src = dataset.data_list
            self.data = np.array([d['path'] for d in src])
            self.targets = np.array([d['label'] for d in src])
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data['image']
        label = data['label']
        return image, label, image


class MammothConcatWrapper(Dataset):
    """
    ConcatDataset(DDR + FGADR)을 Mammoth 호환으로 감싸는 래퍼.
    ConcatDataset은 data_map/data_list 속성이 없으므로 별도 처리.
    """
    def __init__(self, concat_dataset):
        self.dataset = concat_dataset
        
        # Mammoth 필수 속성 구축: targets 배열
        targets = []
        for i in range(len(concat_dataset)):
            # 각 서브 데이터셋의 data_map/data_list에서 라벨 추출
            idx = i
            for ds in concat_dataset.datasets:
                if idx < len(ds):
                    if hasattr(ds, 'data_map'):
                        targets.append(ds.data_map[idx]['label'])
                    elif hasattr(ds, 'data_list'):
                        targets.append(ds.data_list[idx]['label'])
                    else:
                        # fallback: 실제로 로드
                        sample = ds[idx]
                        targets.append(sample['label'] if isinstance(sample['label'], int) 
                                     else sample['label'].item())
                    break
                idx -= len(ds)
        
        self.targets = np.array(targets)
        self.data = np.arange(len(concat_dataset))  # 인덱스를 data로 사용

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data['image']
        label = data['label']
        return image, label, image


# =============================================================================
# QWK 메트릭 유틸리티
# =============================================================================
def compute_qwk(model, loader, device):
    """
    모델 평가 시 QWK 계산.
    Mammoth evaluate() 이후 추가로 호출 가능.
    
    Returns:
        dict: {'accuracy': float, 'qwk': float}
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                continue
            
            images = images.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', None))
                if logits is None:
                    logits = list(outputs.values())[0]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100.0 * correct / total if total > 0 else 0.0
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic') if total > 0 else 0.0
    
    return {'accuracy': acc, 'qwk': qwk}


# =============================================================================
# 메인 데이터셋 클래스
# =============================================================================
class DRGrading(ContinualDataset):
    """
    Mammoth 호환 DR Grading 데이터셋.
    
    [Task 구성]
    Task 0: Base Session
      - 비교 모델(EWC, LwF 등): DDR+FGADR 전체 (classification only)
      - DICAN: Phase 1-A(FGADR only, seg) → 1-B/C(DDR+FGADR, cls)
              → DICAN은 자체 train.py로 학습, 여기서는 비교 모델용
    Task 1: APTOS 2019
    Task 2: Messidor-2
    Task 3: DRAC22
    
    [QWK 지원]
    get_data_loaders() 반환 시 self.test_loaders에 저장하여
    evaluate_with_qwk()로 전체 task QWK 계산 가능.
    """
    NAME = 'dr-grading'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 4
    SIZE = (3, 224, 224)
    
    # ★ QWK 추적용
    _test_loaders = {}

    def get_data_loaders(self):
        task_id = self.current_task
        print(f"\n[Mammoth] Loading Data for Task {task_id}...")

        if task_id == 0:
            # ==================================================
            # Task 0: Base Session
            # ★ DDR + FGADR 전체 사용 (비교 모델용)
            # DICAN은 자체 파이프라인에서 FGADR-only seg를 처리
            # ==================================================
            print(f"[*] Loading Base Dataset: DDR + FGADR")
            print(f"    DDR:   {DATA_ROOT_DDR}")
            print(f"    FGADR: {DATA_ROOT_FGADR}")
            
            ddr_train = DDRBaseDataset(root_dir=DATA_ROOT_DDR, split='train', img_size=224)
            ddr_val = DDRBaseDataset(root_dir=DATA_ROOT_DDR, split='valid', img_size=224)
            
            use_fgadr = os.path.exists(DATA_ROOT_FGADR)
            
            if use_fgadr:
                fgadr_train = FGADRSegDataset(root_dir=DATA_ROOT_FGADR, split='train', img_size=224)
                fgadr_val = FGADRSegDataset(root_dir=DATA_ROOT_FGADR, split='valid', img_size=224)
                
                train_concat = ConcatDataset([ddr_train, fgadr_train])
                test_concat = ConcatDataset([ddr_val, fgadr_val])
                
                train_dataset = MammothConcatWrapper(train_concat)
                test_dataset = MammothConcatWrapper(test_concat)
                
                print(f"    ✅ Combined: Train={len(train_concat)}, Val={len(test_concat)}")
                print(f"       DDR  Train:{len(ddr_train)} Val:{len(ddr_val)}")
                print(f"       FGADR Train:{len(fgadr_train)} Val:{len(fgadr_val)}")
            else:
                print(f"    ⚠️ FGADR not found, using DDR only")
                train_dataset = MammothWrapper(ddr_train)
                test_dataset = MammothWrapper(ddr_val)

        else:
            # ==================================================
            # Task 1+: Incremental Session
            # ==================================================
            session_map = {1: "APTOS 2019", 2: "Messidor-2", 3: "DRAC22"}
            session_name = session_map.get(task_id, f"Session {task_id}")
            
            print(f"[*] Loading Incremental Dataset: {session_name}")
            print(f"    -> Path: {DATA_ROOT_INC}")

            train_raw = UnifiedIncrementalDataset(
                session_id=task_id,
                data_dir=DATA_ROOT_INC,
                img_size=224,
                shot=10,
                split='train'
            )
            test_raw = UnifiedIncrementalDataset(
                session_id=task_id,
                data_dir=DATA_ROOT_INC,
                img_size=224,
                shot=None,
                split='test'
            )
            
            train_dataset = MammothWrapper(train_raw)
            test_dataset = MammothWrapper(test_raw)

        # DataLoader 생성 (Mammoth 유틸리티)
        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)
        
        if hasattr(train_loader, 'num_workers'):
            train_loader.num_workers = 4
        if hasattr(test_loader, 'num_workers'):
            test_loader.num_workers = 4

        # ★ QWK 계산용으로 test_loader 저장
        DRGrading._test_loaders[task_id] = test_loader

        # 라벨 확인 로그
        print(f"\n[*] 🔍 Task {task_id} Labels:")
        train_labels = np.unique(train_dataset.targets)
        print(f"    -> [Train] Unique Labels: {train_labels} (Should be 0-4)")
        print(f"    -> [Train] Size: {len(train_dataset)}")
        print(f"    -> [Test]  Size: {len(test_dataset)}")

        return train_loader, test_loader

    # =================================================================
    # ★ QWK 평가 메서드
    # =================================================================
    @classmethod
    def evaluate_with_qwk(cls, model, device, task_ids=None):
        """
        전체 또는 특정 task에 대해 Accuracy + QWK 계산.
        
        Usage:
            # 학습 후 호출
            results = DRGrading.evaluate_with_qwk(model, device)
            for tid, metrics in results.items():
                print(f"Task {tid}: Acc={metrics['accuracy']:.2f}%, QWK={metrics['qwk']:.4f}")
        
        Args:
            model: 학습된 모델
            device: torch.device
            task_ids: 평가할 task ID 리스트 (None이면 저장된 모든 task)
        
        Returns:
            dict: {task_id: {'accuracy': float, 'qwk': float}}
        """
        if task_ids is None:
            task_ids = sorted(cls._test_loaders.keys())
        
        results = {}
        print(f"\n{'='*55}")
        print(f"  QWK Evaluation (Tasks: {task_ids})")
        print(f"{'='*55}")
        
        for tid in task_ids:
            if tid not in cls._test_loaders:
                print(f"  Task {tid}: No test loader available (skipped)")
                continue
            
            loader = cls._test_loaders[tid]
            metrics = compute_qwk(model, loader, device)
            results[tid] = metrics
            
            print(f"  Task {tid}: Acc={metrics['accuracy']:.2f}%  |  QWK={metrics['qwk']:.4f}")
        
        # 평균
        if results:
            avg_acc = np.mean([m['accuracy'] for m in results.values()])
            avg_qwk = np.mean([m['qwk'] for m in results.values()])
            print(f"  {'─'*50}")
            print(f"  Average: Acc={avg_acc:.2f}%  |  QWK={avg_qwk:.4f}")
        
        print(f"{'='*55}\n")
        return results

    @classmethod
    def get_seg_loaders(cls, batch_size=32, num_workers=4):
        """
        ★ DICAN Phase 1-A 전용: FGADR-only seg loader 반환.
        
        비교 모델(EWC 등)은 이 메서드를 호출하지 않음.
        DICAN의 자체 학습 파이프라인에서만 사용.
        
        Returns:
            dict: {'seg_train': DataLoader, 'seg_val': DataLoader}
                  또는 FGADR 없으면 None
        """
        if not os.path.exists(DATA_ROOT_FGADR):
            print(f"[Warning] FGADR not found: {DATA_ROOT_FGADR}")
            return None
        
        from torch.utils.data import DataLoader
        
        fgadr_train = FGADRSegDataset(root_dir=DATA_ROOT_FGADR, split='train', img_size=224)
        fgadr_val = FGADRSegDataset(root_dir=DATA_ROOT_FGADR, split='valid', img_size=224)
        
        seg_train_loader = DataLoader(
            fgadr_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        seg_val_loader = DataLoader(
            fgadr_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"[*] FGADR Seg Loaders: Train={len(fgadr_train)}, Val={len(fgadr_val)}")
        
        return {
            'seg_train': seg_train_loader,
            'seg_val': seg_val_loader,
        }

    def get_transform(self):
        return transforms.Compose([])

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Compose([])

    @staticmethod
    def get_denormalization_transform():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                                 std=[1/s for s in std])
        ])