import os
from PIL import Image
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import numpy as np
import torch

class DDR_CBM_Dataset(Dataset):
    def __init__(self, split="train", hf_token=None):
        self.repo_id = "ctmedtech/DDR-dataset"
        self.split = split
        self.token = hf_token
        
        # 1. 라벨 파일(txt) 다운로드 및 파싱
        print(f"[{split}] 라벨 파일 로드 중...")
        label_path = hf_hub_download(
            repo_id=self.repo_id, 
            filename=f"DR_grading/{split}.txt", 
            repo_type="dataset",
            token=self.token
        )
        
        self.data_map = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0] # 예: 007-0004-100.jpg
                    label = int(parts[1])
                    img_id = os.path.splitext(img_name)[0] # 확장자 제거 (ID만 추출)
                    self.data_map.append({
                        "id": img_id,
                        "img_name": img_name,
                        "label": label
                    })

        # 2. 병변(Concept) 타입 정의
        self.lesion_types = ["EX", "HE", "MA", "SE"] 
        # 참고: 실제 파일 구조 확인 후 경로 수정 필요 (아래는 일반적인 DDR 구조 가정)
        
    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        img_id = data['id']
        
        # --- A. 이미지 로드 ($X$) ---
        # 로컬 용량이 부족하면 이 부분에서 hf_hub_download로 그때그때 받거나, 
        # 미리 받아둔 폴더에서 읽도록 수정 가능.
        # 여기서는 '필요할 때 다운로드' 하는 방식을 예시로 듭니다.
        try:
            img_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"DR_grading/{self.split}/{data['img_name']}",
                repo_type="dataset",
                token=self.token
            )
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # 이미지가 없는 경우 예외 처리 (검은 이미지 등)
            image = Image.new("RGB", (512, 512))

        # --- B. 마스크 로드 및 Concept 변환 ($C$) ---
        concepts = []
        masks = {}
        
        for lesion in self.lesion_types:
            # *중요*: Step 1에서 확인한 구조에 맞춰 경로 수정 필요
            # 예시 1: 하위 폴더 구조 (lesion_segmentation/train/EX/ID.png)
            # 예시 2: 접미사 구조 (lesion_segmentation/train/ID_EX.png)
            # 여기선 DDR의 흔한 패턴인 '접미사' 방식을 가정해봅니다. (ID_EX.tif 또는 .png)
            mask_filename = f"lesion_segmentation/{self.split}/{img_id}_{lesion}.png"
            
            try:
                # 마스크 파일이 존재하는지 확인하고 다운로드
                mask_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=mask_filename,
                    repo_type="dataset",
                    token=self.token
                )
                mask = Image.open(mask_path).convert("L") # 흑백 변환
                mask_arr = np.array(mask)
                
                # CBM Concept 정의: "해당 병변이 있는가?" (Binary)
                # 픽셀값이 0보다 큰 영역이 조금이라도 있으면 1, 아니면 0
                has_lesion = 1.0 if np.max(mask_arr) > 0 else 0.0
                
            except:
                # 마스크 파일이 없으면 병변 없음으로 간주
                mask = Image.new("L", image.size)
                has_lesion = 0.0
            
            concepts.append(has_lesion)
            masks[lesion] = mask # 시각화용 마스크 저장

        # --- C. 반환 ---
        return {
            "image": image,           # Model Input
            "concepts": torch.tensor(concepts, dtype=torch.float), # CBM Concept Label (EX, HE, MA, SE)
            "label": torch.tensor(data['label'], dtype=torch.long), # Final Task Label (0-4)
            "id": img_id              # 추적용 ID
        }

# 사용 예시
dataset = DDR_CBM_Dataset(split="train")
sample = dataset[500]
print(f"ID: {sample['id']}, Concept(병변유무): {sample['concepts']}, Grade: {sample['label']}")
