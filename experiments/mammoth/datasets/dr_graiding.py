import sys
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

# Mammoth 필수 유틸리티
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders

# -----------------------------------------------------------------------------
# [경로 설정] 프로젝트 루트 및 데이터셋 경로
# -----------------------------------------------------------------------------
PROJECT_ROOT = '/root/DICAN'
DATA_ROOT_DDR = '/root/DICAN_DATASETS/DDR'     # Base Session (DDR)
DATA_ROOT_INC = '/root/DICAN_DATASETS'         # Inc Session (APTOS, etc.)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# DICAN 커스텀 로더 임포트
try:
    from data.base_loader import DDRBaseDataset
    from data.inc_loader import UnifiedIncrementalDataset
except ImportError as e:
    print(f"[Error] DICAN 데이터 로더를 찾을 수 없습니다. 경로를 확인하세요: {PROJECT_ROOT}")
    raise e

# 기존 import 아래에 추가
import torch

class MammothWrapper(Dataset):
    """
    [Standard Model Compatible Wrapper]
    EWC, ER 등 일반 Mammoth 모델을 위한 래퍼입니다.
    1. 반환값: (image, label, original_image) -> 총 3개 필수!
    2. 입력형태: 순수 3채널 RGB 이미지 (마스크 제거)
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
        # Mammoth 필수 속성 (Data Splitting용)
        if hasattr(dataset, 'data_map'): # Base (DDR)
            src = dataset.data_map
            self.data = np.array([d['img_name'] for d in src])
            self.targets = np.array([d['label'] for d in src])
        elif hasattr(dataset, 'data_list'): # Inc (APTOS...)
            src = dataset.data_list
            self.data = np.array([d['path'] for d in src])
            self.targets = np.array([d['label'] for d in src])
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # 1. 이미지 (Tensor: [3, 224, 224])
        image = data['image']
        
        # 2. 라벨 (Int/Long)
        label = data['label']
        
        # [핵심 수정 1] 마스크 제거 (Standard 모델은 7채널 입력을 못 받음)
        # EWC는 Classification 모델이므로 마스크가 필요 없습니다.
        
        # [핵심 수정 2] 3번째 인자 반환 (Mammoth 요구사항)
        # Mammoth는 내부적으로 (input, label, not_aug_input)을 기대합니다.
        # 별도의 Augmentation이 없다면 image를 그대로 한 번 더 반환하면 해결됩니다.
        
        return image, label, image

class DRGrading(ContinualDataset):
    """
    Continual Learning을 위한 메인 데이터셋 클래스 (Mammoth 호환)
    Task 0: DDR (Base Session, with Masks)
    Task 1+: APTOS, Messidor-2, DRAC22 (Incremental Session, No Masks)
    """
    NAME = 'dr-grading'
    SETTING = 'domain-il'  # Domain-Incremental Learning
    N_CLASSES_PER_TASK = 5 
    N_TASKS = 4 
    SIZE = (3, 224, 224)

    def get_data_loaders(self):
        task_id = self.current_task
        print(f"\n[Mammoth] Loading Data for Task {task_id}...")

        if task_id == 0:
            # ==========================================
            # Task 0: Base Session (DDR) - Local Load
            # ==========================================
            print(f"[*] Loading DDR Dataset from: {DATA_ROOT_DDR}")
            
            # base_loader.py의 DDRBaseDataset 사용
            # [주의] Test Transform은 Mammoth 내부에서 처리되거나 로더 안에서 처리됨.
            # 여기서는 로더가 이미 Transform을 포함하고 있으므로 그대로 사용.
            train_raw = DDRBaseDataset(root_dir=DATA_ROOT_DDR, split='train', img_size=224)
            test_raw = DDRBaseDataset(root_dir=DATA_ROOT_DDR, split='valid', img_size=224)

        else:
            # ==========================================
            # Task 1+: Incremental Session
            # ==========================================
            # Task ID 매핑 (1 -> APTOS, 2 -> Messidor, 3 -> DRAC)
            # UnifiedIncrementalDataset은 session_id 1, 2, 3을 받음
            session_map = {1: "APTOS 2019", 2: "Messidor-2", 3: "DRAC22"}
            session_name = session_map.get(task_id, f"Session {task_id}")
            
            print(f"[*] Loading Incremental Dataset: {session_name}")
            print(f"    -> Path: {DATA_ROOT_INC}")

            # inc_loader.py의 UnifiedIncrementalDataset 사용
            train_raw = UnifiedIncrementalDataset(
                session_id=task_id,
                data_dir=DATA_ROOT_INC,
                img_size=224,
                shot=10,        # Few-shot 설정 (필요시 args에서 받아오게 수정 가능)
                split='train'
            )
            # Test 셋은 Shot 제한 없이 전체 사용
            test_raw = UnifiedIncrementalDataset(
                session_id=task_id,
                data_dir=DATA_ROOT_INC,
                img_size=224,
                shot=None, 
                split='test'    # 또는 validation 구조에 따라 'val' 사용
            )

        # 2. Mammoth 호환 래퍼 적용
        train_dataset = MammothWrapper(train_raw)
        test_dataset = MammothWrapper(test_raw)

        # 3. DataLoader 생성 (Mammoth 유틸리티 사용)
        # store_masked_loaders는 내부적으로 Class Incremental 등의 마스킹을 처리하지만,
        # Domain-IL 설정에서는 전체 클래스를 다 보여주도록 동작함.
        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)

        # [안전장치] 로더의 워커 수 조정 (로컬 파일이므로 4~8 적당)
        # Mammoth가 내부적으로 생성한 로더 속성을 덮어씌움
        if hasattr(train_loader, 'num_workers'): train_loader.num_workers = 4
        if hasattr(test_loader, 'num_workers'): test_loader.num_workers = 4

        # [확인용 로그]
        print(f"\n[*] 🔍 Checking Labels for Task {task_id}...")
        train_labels = np.unique(train_dataset.targets)
        print(f"    -> [Train] Unique Labels: {train_labels} (Should be 0-4)")

        return train_loader, test_loader

    def get_transform(self):
        # 데이터셋 내부에서 이미 transform을 수행하므로, 
        # Mammoth가 추가적으로 transform을 적용하지 않도록 Identity 반환
        return transforms.Compose([])

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        # 역시 데이터셋 내부에서 Normalize까지 끝난 상태임.
        return transforms.Compose([])
    
    @staticmethod
    def get_denormalization_transform():
        # 시각화 등을 위해 역변환 필요시 사용
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                                 std=[1/s for s in std])
        ])