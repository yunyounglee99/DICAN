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
        
        # [수정 포인트] 5번 라벨 필터링 로직
        discard_count = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    
                    # 5번(Ungradable) 라벨인 경우 리스트에 담지 않고 건너뜁니다.
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
            
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def _load_masks(self, img_id, original_size):
        """
        DICAN의 핵심: 4가지 병변 마스크를 찾아 4채널 텐서로 반환
        경로: lesion_segmentation/annotations/{seg_split}/{img_id}_{lesion}.ext
        """
        mask_stack = []
        w, h = original_size
        
        # 마스크가 있는 폴더 경로 (스크린샷 기반)
        mask_root = os.path.join(self.root_dir, "lesion_segmentation", "annotations", self.seg_split)
        
        for lesion in self.lesion_types:
            found_path = None
            # 확장자 유연하게 탐색 (DDR은 보통 .tif지만 변환되었을 수 있음)
            for ext in ["tif", "png", "jpg", "TIF", "PNG"]:
                # 파일명 패턴: ID_EX.tif 형태 가정 (DDR 표준)
                # 만약 ID 폴더 안에 있는 구조라면 os.path.join(mask_root, lesion, ...) 수정 필요
                # 여기서는 가장 일반적인 'annotations/train/007-0004-100_EX.tif' 패턴 가정
                candidate = os.path.join(mask_root, f"{img_id}_{lesion}.{ext}")
                
                if os.path.exists(candidate):
                    found_path = candidate
                    break
            
            if found_path:
                # 마스크 존재: 로드 -> 흑백변환 -> 텐서
                try:
                    mask = Image.open(found_path).convert("L")
                    # 마스크 리사이징 (Nearest Neighbor로 정보 보존)
                    mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
                    mask_np = np.array(mask)
                    # 0보다 크면 1, 아니면 0 (Binary Mask)
                    mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32))
                except Exception as e:
                    print(f"[Warning] Failed to load mask {found_path}: {e}")
                    mask_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            else:
                # 마스크 부재: 해당 병변이 없는 이미지 (All-Zero Tensor)
                mask_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            
            mask_stack.append(mask_tensor)
            
        # [4, H, W] 형태로 병합
        return torch.stack(mask_stack, dim=0)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        
        # 1. 이미지 로드
        img_path = os.path.join(self.root_dir, "DR_grading", self.grad_split, data['img_name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # 혹시 모를 에러 방지를 위해 예외 처리
            print(f"[Error] Image not found: {img_path}")
            image = Image.new("RGB", (self.img_size, self.img_size))

        w, h = image.size

        # 2. 마스크 로드 (DICAN용 4채널 마스크)
        masks = self._load_masks(data['id'], (w, h))

        # 3. Transform
        # 이미지: Bilinear Resize -> Tensor -> Normalize
        image = TF.resize(image, (self.img_size, self.img_size))
        image = TF.to_tensor(image)
        image = self.normalize(image)

        return {
            "image": image, 
            "masks": masks, 
            "label": torch.tensor(data['label'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data_map)

if __name__ == "__main__":
    import sys
    
    # -------------------------------------------------------------------------
    # [설정] 실제 데이터셋이 있는 경로로 맞춰주세요.
    # 아까 100GB 받으신 경로: /root/DICAN_DATASETS/DDR
    # -------------------------------------------------------------------------
    DATA_ROOT = '/Volumes/Nyoungs_SSD/macbook/dev/datasets/DICAN_DATASETS/DDR'
    
    print(f"[*] Testing DDRBaseDataset with root: {DATA_ROOT}\n")
    
    if not os.path.exists(DATA_ROOT):
        print(f"[Error] Directory not found: {DATA_ROOT}")
        print("Set 'DATA_ROOT' correctly in the __main__ block.")
        sys.exit(1)

    # 1. Train Set 테스트
    try:
        print("="*60)
        print("[Test 1] Loading Train Split...")
        train_ds = DDRBaseDataset(root_dir=DATA_ROOT, split='train')
        
        print(f"  -> Train Size: {len(train_ds)}")
        
        # 첫 번째 샘플 가져오기
        sample = train_ds[0]
        img = sample['image']
        mask = sample['masks']
        label = sample['label']
        
        print(f"  -> Sample[0] Image Shape: {img.shape} (3, 224, 224 expected)")
        print(f"  -> Sample[0] Mask Shape : {mask.shape} (4, 224, 224 expected)")
        print(f"  -> Sample[0] Label      : {label} (0~4)")
        print(f"  -> Mask Unique Values   : {torch.unique(mask)} (should be 0. and 1.)")
        
    except Exception as e:
        print(f"  -> [FAIL] Train Load Error: {e}")
        import traceback
        traceback.print_exc()

    # 2. Valid Set 테스트 (경로 매핑 확인)
    try:
        print("\n" + "="*60)
        print("[Test 2] Loading Valid Split (Check Folder Mapping)...")
        # 여기서 split='valid'를 넣었을 때 내부에서 'val' 폴더를 잘 찾는지 확인
        valid_ds = DDRBaseDataset(root_dir=DATA_ROOT, split='valid')
        
        print(f"  -> Valid Size: {len(valid_ds)}")
        
        # 중간쯤 있는 샘플 가져오기 (랜덤하게 병변이 있을만한 것)
        idx = len(valid_ds) // 2
        sample = valid_ds[idx]
        
        print(f"  -> Sample[{idx}] Image Shape: {sample['image'].shape}")
        print(f"  -> Sample[{idx}] Mask Shape : {sample['masks'].shape}")
        
        # 마스크가 실제로 들어있는지 확인 (All zero가 아닌지)
        if sample['masks'].sum() > 0:
            print("  -> [Success] Active lesion found in mask!")
        else:
            print("  -> [Note] This sample has no lesions (Normal or mask missing).")

    except Exception as e:
        print(f"  -> [FAIL] Valid Load Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*60)
    print("Done.")