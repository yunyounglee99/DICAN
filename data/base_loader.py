"""
DDR Base Dataset Loader
========================
[버그 수정]
기존: annotations/train/{img_id}_{lesion}.tif  ← 파일 못 찾음 → 전부 zero
수정: annotations/train/{LESION}/{img_id}.tif  ← DDR 실제 구조

[추가]
- has_mask 플래그: 마스크 존재 여부를 미리 캐싱
- FGADR ConcatDataset 호환 출력 포맷
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class DDRBaseDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224):
        self.img_size = img_size
        self.split = split
        self.root_dir = root_dir 
        self.lesion_types = ["EX", "HE", "MA", "SE"]

        self.grad_split = split
        self.seg_split = 'val' if split == 'valid' else split

        print(f"[*] Loading DDR ({split}) from local disk: {self.root_dir}")

        label_path = os.path.join(self.root_dir, "DR_grading", f"{split}.txt")
        self.data_map = []
        
        discard_count = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    
                    if label == 5:
                        discard_count += 1
                        continue
                        
                    img_id = os.path.splitext(img_name)[0]
                    self.data_map.append({
                        "id": img_id,
                        "img_name": img_name,
                        "label": label
                    })
        
        print(f"    -> {len(self.data_map)} samples loaded.")
        if discard_count > 0:
            print(f"    -> [Note] Discarded {discard_count} samples with label 5 (Ungradable).")
        
        # ★ 마스크 존재 여부 사전 캐싱
        self._cache_mask_availability()
            
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def _cache_mask_availability(self):
        """
        마스크 존재 여부 미리 확인.
        DDR: Grading(6835) >> Segmentation(383), 겹치는 ID 275개만 마스크 있음.
        """
        mask_root = os.path.join(
            self.root_dir, "lesion_segmentation", "annotations", self.seg_split
        )
        
        has_mask_count = 0
        for item in self.data_map:
            has_any = False
            for lesion in self.lesion_types:
                # ★ 수정된 경로: {split}/{LESION}/{img_id}.{ext}
                for ext in ["tif", "png", "TIF", "PNG"]:
                    candidate = os.path.join(mask_root, lesion, f"{item['id']}.{ext}")
                    if os.path.exists(candidate):
                        has_any = True
                        break
                if has_any:
                    break
            
            item['has_mask'] = has_any
            if has_any:
                has_mask_count += 1
        
        print(f"    -> Mask available: {has_mask_count}/{len(self.data_map)} images "
              f"({100*has_mask_count/max(len(self.data_map),1):.1f}%)")

    def _load_masks(self, img_id, original_size):
        """
        4가지 병변 마스크를 [4, 224, 224] 텐서로 반환.
        
        ★ 수정: DDR 실제 구조 반영
        기존 (틀림): annotations/{split}/{img_id}_{LESION}.tif
        수정 (맞음): annotations/{split}/{LESION}/{img_id}.tif
        """
        mask_stack = []
        
        mask_root = os.path.join(
            self.root_dir, "lesion_segmentation", "annotations", self.seg_split
        )
        
        for lesion in self.lesion_types:
            found_path = None
            # ★ 수정된 경로 패턴: {split}/{LESION}/{img_id}.{ext}
            for ext in ["tif", "png", "TIF", "PNG"]:
                candidate = os.path.join(mask_root, lesion, f"{img_id}.{ext}")
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
                    mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32))
                except Exception as e:
                    print(f"[Warning] Failed to load mask {found_path}: {e}")
                    mask_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            else:
                mask_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            
            mask_stack.append(mask_tensor)
            
        return torch.stack(mask_stack, dim=0)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        
        # 1. 이미지 로드
        img_path = os.path.join(self.root_dir, "DR_grading", self.grad_split, data['img_name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Error] Image not found: {img_path}")
            image = Image.new("RGB", (self.img_size, self.img_size))

        w, h = image.size

        # 2. 마스크 로드
        masks = self._load_masks(data['id'], (w, h))

        # 3. Transform
        image = TF.resize(image, (self.img_size, self.img_size))
        image = TF.to_tensor(image)
        image = self.normalize(image)

        # ★ FGADR과 동일한 포맷 (ConcatDataset 호환)
        return {
            "image": image, 
            "masks": masks, 
            "label": torch.tensor(data['label'], dtype=torch.long),
            "has_mask": data['has_mask']
        }

    def __len__(self):
        return len(self.data_map)


if __name__ == "__main__":
    import sys
    
    DATA_ROOT = '/root/DICAN_DATASETS/DDR'
    
    print(f"[*] Testing DDRBaseDataset with root: {DATA_ROOT}\n")
    
    if not os.path.exists(DATA_ROOT):
        print(f"[Error] Directory not found: {DATA_ROOT}")
        sys.exit(1)

    print("=" * 60)
    print("[Test 1] Loading Train Split...")
    train_ds = DDRBaseDataset(root_dir=DATA_ROOT, split='train')
    print(f"  -> Train Size: {len(train_ds)}")
    
    # 마스크가 있는 샘플 찾기
    mask_found = False
    for i in range(len(train_ds)):
        sample = train_ds[i]
        if sample['masks'].sum() > 0:
            print(f"\n  -> Sample[{i}] WITH mask:")
            print(f"     Image: {sample['image'].shape}")
            print(f"     Masks: {sample['masks'].shape}")
            print(f"     Label: {sample['label']}")
            print(f"     has_mask: {sample['has_mask']}")
            print(f"     Mask unique: {torch.unique(sample['masks'])}")
            for k, name in enumerate(["EX", "HE", "MA", "SE"]):
                px = (sample['masks'][k] > 0).sum().item()
                print(f"     {name}: {px} active pixels")
            mask_found = True
            break
    
    if not mask_found:
        print("  -> [ERROR] No masks found in entire dataset!")
    
    sample0 = train_ds[0]
    print(f"\n  -> Sample[0]: mask_sum={sample0['masks'].sum():.0f}, has_mask={sample0['has_mask']}")

    print("\n" + "=" * 60)
    print("[Test 2] Loading Valid Split...")
    valid_ds = DDRBaseDataset(root_dir=DATA_ROOT, split='valid')
    print(f"  -> Valid Size: {len(valid_ds)}")
    
    print("\n" + "=" * 60)
    print("Done.")