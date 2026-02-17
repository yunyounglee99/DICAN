"""
FGADR Seg-set Dataset Loader
==============================
FGADR은 1842개 이미지 전부에 4종 병변 마스크 + DR 등급 라벨 보유.
DDR(275개 마스크)을 보완하여 Seg Loss 학습량을 6배 증가시킴.

[FGADR 구조]
FGADR/
  Seg-set/
    Original_Images/          ← 원본 이미지 (.png)
    HardExudate_Masks/        → EX
    Hemohedge_Masks/          → HE
    Microaneurysms_Masks/     → MA
    SoftExudate_Masks/        → SE
    IRMA_Masks/               → 무시 (DICAN 4 concept에 불포함)
    Neovascularization_Masks/ → 무시
    DR_Seg_Grading_Label.csv  ← 이미지명, DR등급

[DDR과의 호환성]
__getitem__ 반환값이 DDR과 동일한 dict 형태:
  {"image": [3,224,224], "masks": [4,224,224], "label": long, "has_mask": bool}
→ torch.utils.data.ConcatDataset으로 DDR과 바로 합칠 수 있음.
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class FGADRSegDataset(Dataset):
    """
    FGADR Seg-set 데이터셋.
    DDR BaseDataset과 동일한 출력 포맷을 유지하여 ConcatDataset 호환.
    """
    
    # FGADR 폴더명 → DICAN Concept 매핑
    LESION_FOLDER_MAP = {
        "EX": "HardExudate_Masks",
        "HE": "Hemohedge_Masks",
        "MA": "Microaneurysms_Masks",
        "SE": "SoftExudate_Masks",
    }
    
    def __init__(self, root_dir, split="train", img_size=224, 
                 val_ratio=0.15, seed=42):
        """
        Args:
            root_dir: FGADR 루트 (예: /root/DICAN_DATASETS/FGADR)
            split: "train" or "valid" (FGADR에는 공식 split이 없으므로 내부 분할)
            img_size: 리사이즈 크기
            val_ratio: validation 비율
            seed: 재현성을 위한 시드
        """
        self.img_size = img_size
        self.split = split
        self.root_dir = root_dir
        self.seg_root = os.path.join(root_dir, "Seg-set")
        self.img_dir = os.path.join(self.seg_root, "Original_Images")
        self.lesion_types = ["EX", "HE", "MA", "SE"]
        
        print(f"[*] Loading FGADR Seg-set ({split}) from: {self.seg_root}")
        
        # 1. CSV에서 이미지 목록 + DR 등급 로드
        self.data_map = self._load_csv()
        
        # 2. CSV가 없거나 비어있으면 폴더 스캔으로 대체
        if not self.data_map:
            print("    -> CSV 로드 실패, 폴더 스캔으로 대체...")
            self.data_map = self._load_from_folder()
        
        # 3. Label 5 (Ungradable) 필터링
        before = len(self.data_map)
        self.data_map = [d for d in self.data_map if d['label'] != 5]
        discarded = before - len(self.data_map)
        if discarded > 0:
            print(f"    -> Discarded {discarded} ungradable (label=5) samples.")
        
        # 4. Train/Valid 분할
        self._split_dataset(val_ratio, seed)
        
        # 5. 마스크 존재 여부 확인
        self._verify_masks()
        
        print(f"    -> {len(self.data_map)} samples loaded for '{split}'.")
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _load_csv(self):
        """DR_Seg_Grading_Label.csv 파싱"""
        csv_path = os.path.join(self.seg_root, "DR_Seg_Grading_Label.csv")
        
        if not os.path.exists(csv_path):
            print(f"    -> CSV not found: {csv_path}")
            return []
        
        data_map = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            return []
        
        # 헤더 감지 (첫 행이 숫자가 아니면 헤더)
        header = rows[0]
        start_idx = 0
        try:
            int(header[-1])
            # 첫 행이 데이터 (헤더 없음)
            start_idx = 0
        except (ValueError, IndexError):
            # 첫 행이 헤더
            start_idx = 1
            print(f"    -> CSV 컬럼: {header}")
        
        for row in rows[start_idx:]:
            if len(row) < 2:
                continue
            
            img_name = row[0].strip()
            try:
                label = int(row[-1].strip())  # 마지막 컬럼 = DR 등급
            except ValueError:
                continue
            
            # 이미지 ID 추출 (확장자 제거)
            img_id = os.path.splitext(img_name)[0]
            
            # 확장자가 없으면 자동 탐색
            if not os.path.splitext(img_name)[1]:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    candidate = os.path.join(self.img_dir, img_id + ext)
                    if os.path.exists(candidate):
                        img_name = img_id + ext
                        break
            
            data_map.append({
                "id": img_id,
                "img_name": img_name,
                "label": label
            })
        
        print(f"    -> CSV: {len(data_map)} entries parsed.")
        return data_map

    def _load_from_folder(self):
        """CSV 없을 때 Original_Images 폴더에서 직접 로드 (라벨 없음 → -1)"""
        if not os.path.exists(self.img_dir):
            print(f"    -> [ERROR] Image dir not found: {self.img_dir}")
            return []
        
        data_map = []
        for fname in sorted(os.listdir(self.img_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_id = os.path.splitext(fname)[0]
                data_map.append({
                    "id": img_id,
                    "img_name": fname,
                    "label": -1  # 라벨 불명
                })
        
        print(f"    -> Folder scan: {len(data_map)} images (labels unknown)")
        return data_map

    def _split_dataset(self, val_ratio, seed):
        """Train/Valid 분할 (FGADR에는 공식 split이 없으므로)"""
        import random
        rng = random.Random(seed)
        
        indices = list(range(len(self.data_map)))
        rng.shuffle(indices)
        
        n_val = int(len(indices) * val_ratio)
        
        if self.split == 'valid' or self.split == 'val':
            selected = sorted(indices[:n_val])
        else:  # train
            selected = sorted(indices[n_val:])
        
        self.data_map = [self.data_map[i] for i in selected]

    def _verify_masks(self):
        """마스크 파일 존재 확인 + has_mask 플래그"""
        has_mask_count = 0
        
        for item in self.data_map:
            has_any = False
            for lesion in self.lesion_types:
                folder = self.LESION_FOLDER_MAP[lesion]
                mask_dir = os.path.join(self.seg_root, folder)
                
                for ext in ['.png', '.bmp', '.tif', '.jpg']:
                    candidate = os.path.join(mask_dir, item['id'] + ext)
                    if os.path.exists(candidate):
                        has_any = True
                        break
                if has_any:
                    break
            
            item['has_mask'] = has_any
            if has_any:
                has_mask_count += 1
        
        print(f"    -> Mask available: {has_mask_count}/{len(self.data_map)} "
              f"({100*has_mask_count/max(len(self.data_map),1):.1f}%)")

    def _load_masks(self, img_id):
        """4가지 병변 마스크를 [4, 224, 224] 텐서로 반환"""
        mask_stack = []
        
        for lesion in self.lesion_types:
            folder = self.LESION_FOLDER_MAP[lesion]
            mask_dir = os.path.join(self.seg_root, folder)
            
            found_path = None
            for ext in ['.png', '.bmp', '.tif', '.jpg']:
                candidate = os.path.join(mask_dir, img_id + ext)
                if os.path.exists(candidate):
                    found_path = candidate
                    break
            
            if found_path:
                try:
                    mask = Image.open(found_path).convert("L")
                    mask = TF.resize(
                        mask, (self.img_size, self.img_size),
                        interpolation=transforms.InterpolationMode.NEAREST
                    )
                    mask_np = np.array(mask)
                    # 0보다 크면 1 (FGADR 마스크가 0/255일 수 있음)
                    mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32))
                except Exception as e:
                    print(f"[Warning] Failed to load FGADR mask {found_path}: {e}")
                    mask_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            else:
                mask_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            
            mask_stack.append(mask_tensor)
        
        return torch.stack(mask_stack, dim=0)  # [4, 224, 224]

    def __getitem__(self, idx):
        data = self.data_map[idx]
        
        # 1. 이미지 로드
        img_path = os.path.join(self.img_dir, data['img_name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Error] FGADR image not found: {img_path}")
            image = Image.new("RGB", (self.img_size, self.img_size))
        
        # 2. 마스크 로드
        masks = self._load_masks(data['id'])
        
        # 3. Transform
        image = TF.resize(image, (self.img_size, self.img_size))
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # ★ DDR과 동일한 dict 포맷 → ConcatDataset 호환
        return {
            "image": image,                                              # [3, 224, 224]
            "masks": masks,                                              # [4, 224, 224]
            "label": torch.tensor(data['label'], dtype=torch.long),      # scalar
            "has_mask": data['has_mask']                                  # bool
        }

    def __len__(self):
        return len(self.data_map)


# =================================================================
# 테스트
# =================================================================
if __name__ == "__main__":
    import sys
    
    FGADR_ROOT = '/root/DICAN_DATASETS/FGADR'
    
    print(f"[*] Testing FGADRSegDataset with root: {FGADR_ROOT}\n")
    
    if not os.path.exists(FGADR_ROOT):
        print(f"[Error] Directory not found: {FGADR_ROOT}")
        sys.exit(1)
    
    # Train 로드
    print("=" * 60)
    print("[Test 1] Loading FGADR Train Split...")
    train_ds = FGADRSegDataset(root_dir=FGADR_ROOT, split='train')
    print(f"  -> Train Size: {len(train_ds)}")
    
    # 첫 번째 샘플
    sample = train_ds[0]
    print(f"  -> Image: {sample['image'].shape}")
    print(f"  -> Masks: {sample['masks'].shape}")
    print(f"  -> Label: {sample['label'].item()}")
    print(f"  -> has_mask: {sample['has_mask']}")
    print(f"  -> Mask unique: {torch.unique(sample['masks'])}")
    
    # 마스크 통계
    if sample['masks'].sum() > 0:
        for k, name in enumerate(["EX", "HE", "MA", "SE"]):
            px = (sample['masks'][k] > 0).sum().item()
            print(f"     {name}: {px} active pixels")
    else:
        print("  -> [NOTE] First sample has empty masks, trying more...")
        for i in range(min(20, len(train_ds))):
            s = train_ds[i]
            if s['masks'].sum() > 0:
                print(f"  -> Sample[{i}] has active masks!")
                for k, name in enumerate(["EX", "HE", "MA", "SE"]):
                    px = (s['masks'][k] > 0).sum().item()
                    print(f"     {name}: {px} active pixels")
                break
    
    # Valid 로드
    print("\n" + "=" * 60)
    print("[Test 2] Loading FGADR Valid Split...")
    val_ds = FGADRSegDataset(root_dir=FGADR_ROOT, split='valid')
    print(f"  -> Valid Size: {len(val_ds)}")
    
    # 라벨 분포
    print("\n[Test 3] Label Distribution:")
    from collections import Counter
    labels = [train_ds.data_map[i]['label'] for i in range(len(train_ds))]
    dist = Counter(labels)
    for g in sorted(dist.keys()):
        print(f"  Grade {g}: {dist[g]}")
    
    print("\n" + "=" * 60)
    print("Done.")