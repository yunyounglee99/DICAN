import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download

class DDRBaseDataset(Dataset):
    """
    [Base Session용 데이터셋 - 최종 수정]
    - Level 1~4: 마스크 로드 -> 프로토타입 생성 & 분류 학습
    - Level 0: 빈 마스크(Zero Tensor) 로드 -> 분류 학습 & False Positive 억제 학습
    """
    def __init__(self, split="train", hf_token=None, img_size=224):
        self.repo_id = "ctmedtech/DDR-dataset"
        self.split = split
        self.token = hf_token
        self.img_size = img_size
        self.lesion_types = ["EX", "HE", "MA", "SE"]

        print(f"[*] Loading DDR Base Dataset ({split})...")
        
        # 1. 라벨 파일 로드
        try:
            label_path = hf_hub_download(
                repo_id=self.repo_id, 
                filename=f"DR_grading/{split}.txt", 
                repo_type="dataset",
                token=self.token
            )
        except Exception as e:
            raise FileNotFoundError(f"Failed to download label file: {e}")

        self.data_map = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    img_id = os.path.splitext(img_name)[0]
                    
                    # 모든 데이터를 다 포함합니다 (Level 0 포함)
                    self.data_map.append({
                        "id": img_id,
                        "img_name": img_name,
                        "label": label
                    })
        
        print(f"    -> Total samples: {len(self.data_map)} (including Level 0)")

        # 2. 전처리
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_map)

    def _load_image(self, img_name):
        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"DR_grading/{self.split}/{img_name}",
            repo_type="dataset",
            token=self.token
        )
        return Image.open(path).convert("RGB")

    def _load_masks(self, img_id, original_size):
        """마스크가 있으면 로드하고, 없으면(Level 0 등) 0으로 채운 텐서 반환"""
        mask_stack = []
        w, h = original_size

        for lesion in self.lesion_types:
            mask_path = None
            # 확장자 탐색
            for ext in ["tif", "png", "jpg"]:
                try:
                    filename = f"lesion_segmentation/{self.split}/{img_id}_{lesion}.{ext}"
                    # local_files_only=True로 먼저 체크하면 속도가 빠름
                    mask_path = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=filename,
                        repo_type="dataset",
                        token=self.token
                    )
                    break 
                except:
                    continue
            
            if mask_path:
                # 마스크 존재: 로드 후 이진화
                mask = Image.open(mask_path).convert("L")
                mask = TF.resize(mask, (h, w), interpolation=transforms.InterpolationMode.NEAREST)
                mask_np = np.array(mask)
                mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32))
            else:
                # 마스크 부재: All-Zero Tensor (학습에 방해 안 됨)
                mask_tensor = torch.zeros((h, w), dtype=torch.float32)
            
            mask_stack.append(mask_tensor)

        return torch.stack(mask_stack, dim=0)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        
        # 1. 이미지 로드
        image = self._load_image(data['img_name'])
        w, h = image.size
        
        # 2. 마스크 로드 (Level 0는 0-Tensor가 됨)
        masks = self._load_masks(data['id'], (w, h))

        # 3. Transform
        image = TF.resize(image, (self.img_size, self.img_size))
        masks = TF.resize(masks, (self.img_size, self.img_size), 
                          interpolation=transforms.InterpolationMode.NEAREST)
        
        image = TF.to_tensor(image)
        image = self.normalize(image)

        return {
            "image": image,
            "masks": masks,
            "label": torch.tensor(data['label'], dtype=torch.long),
            "id": data['id']
        }

def get_base_loader(batch_size=32, split='train', num_workers=4, hf_token=None):
    dataset = DDRBaseDataset(split=split, hf_token=hf_token)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader

# --- 테스트 코드 ---
if __name__ == "__main__":
    # 토큰이 필요하다면 여기에 입력
    loader = get_base_loader(batch_size=2, split='train')
    
    print("Fetching one batch...")
    batch = next(iter(loader))
    
    images = batch['image']
    masks = batch['masks']
    labels = batch['label']
    
    print(f"Image Shape: {images.shape}") # Expected: [2, 3, 224, 224]
    print(f"Masks Shape: {masks.shape}")  # Expected: [2, 4, 224, 224]
    print(f"Labels: {labels}")
    
    # 마스크가 제대로 로드되었는지(전부 0이 아닌지) 확인
    print(f"Max Mask Value: {masks.max().item()}")