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
        self.root_dir = root_dir # 로컬 캐시 경로
        self.lesion_types = ["EX", "HE", "MA", "SE"]

        # 1. 라벨 파일 로드
        label_path = os.path.join(self.root_dir, "DR_grading", f"{split}.txt")
        self.data_map = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.data_map.append({
                        "id": os.path.splitext(parts[0])[0],
                        "img_name": parts[0],
                        "label": int(parts[1])
                    })
        print(f"    -> {len(self.data_map)} samples loaded from local disk.")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        data = self.data_map[idx]
        # 로컬에서 이미지 로드
        img_path = os.path.join(self.root_dir, "DR_grading", self.split, data['img_name'])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # [DICAN 핵심] 로컬에서 마스크 로드
        mask_stack = []
        for lesion in self.lesion_types:
            found = False
            for ext in ["tif", "png", "jpg"]:
                p = os.path.join(self.root_dir, "lesion_segmentation", self.split, f"{data['id']}_{lesion}.{ext}")
                if os.path.exists(p):
                    mask = Image.open(p).convert("L")
                    mask = TF.resize(mask, (h, w), interpolation=transforms.InterpolationMode.NEAREST)
                    mask_stack.append(torch.from_numpy((np.array(mask) > 0).astype(np.float32)))
                    found = True
                    break
            if not found:
                mask_stack.append(torch.zeros((h, w), dtype=torch.float32))
        
        masks = torch.stack(mask_stack, dim=0)

        # 전처리
        image = TF.resize(image, (self.img_size, self.img_size))
        masks = TF.resize(masks, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = self.normalize(image)

        return {"image": image, "masks": masks, "label": torch.tensor(data['label'], dtype=torch.long)}

    def __len__(self):
        return len(self.data_map)