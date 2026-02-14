import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class UnifiedIncrementalDataset(Dataset):
    """
    [Incremental Session용 통합 데이터셋]
    각 세션(병원/도메인)별로 서로 다른 데이터 소스를 하나의 인터페이스로 통합합니다.
    """
    def __init__(self, session_id, data_dir, img_size=224, shot=10, split='train'):
        """
        Args:
            session_id (int): 1~5 (세션 번호)
            data_dir (str): 해당 데이터셋이 저장된 로컬 루트 경로
            img_size (int): 리사이징 크기
            shot (int): Few-shot 학습을 위한 샘플 수 (None이면 전체 사용)
        """
        self.session_id = session_id
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.data_list = []
        
        # 1. 세션별 데이터 로드 로직
        if session_id == 1: # DRAC22_RNV
            self._load_drac22()
        elif session_id == 2: # RFMiD
            self._load_rfmid()
        elif session_id == 3: # PRIME-FP20 (Local)
            self._load_prime_fp20()
        elif session_id == 4: # APTOS 2019
            self._load_aptos()
        elif session_id == 5: # EYEPACS
            self._load_eyepacs()
        
        # 2. Few-shot 샘플링 (K-shot)
        if shot is not None and split == 'train':
            self._apply_few_shot(shot)

        # 3. 전처리 (ImageNet 기준)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _apply_few_shot(self, shot):
        """등급별로 K-shot만큼 균등하게 샘플링"""
        df = pd.DataFrame(self.data_list)
        few_shot_list = []
        for g in range(5): # 0~4 등급
            subset = df[df['label'] == g]
            if len(subset) > 0:
                sampled = subset.sample(n=min(shot, len(subset)), random_state=42)
                few_shot_list.extend(sampled.to_dict('records'))
        self.data_list = few_shot_list
        print(f"   -> Few-shot applied: Total {len(self.data_list)} samples for Session {self.session_id}")

    # --- 각 데이터셋별 파싱 로직 ---
    def _load_drac22(self):
        # DRAC22 구조에 맞게 수정 (보통 csv 라벨 파일 존재)
        csv_path = os.path.join(self.data_dir, "DRAC22_Grading_Labels.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.data_list.append({
                "path": os.path.join(self.data_dir, "images", row['image_name']),
                "label": int(row['grade'])
            })

    def _load_rfmid(self):
        # RFMiD는 다중 질환 데이터셋이므로 DR 등급만 추출
        csv_path = os.path.join(self.data_dir, "Training_Labels.csv")
        df = pd.read_csv(csv_path)
        # DR 관련 컬럼(예: 'DR')이 1인 경우만 가져오거나 등급 매핑 필요
        for _, row in df.iterrows():
            self.data_list.append({
                "path": os.path.join(self.data_dir, "images", f"{row['ID']}.png"),
                "label": int(row['DR_Grade']) # 데이터셋 컬럼명에 맞춰 수정
            })

    def _load_prime_fp20(self):
        # 로컬 저장된 PRIME-FP20 로드 (UWF 이미지)
        img_dir = os.path.join(self.data_dir, "images")
        for img_name in os.listdir(img_dir):
            if img_name.endswith(('.jpg', '.png')):
                # 파일명에서 라벨 추출하는 로직 또는 별도 txt/csv 로드
                # 예: ID_grade.jpg 형태 가정
                label = int(img_name.split('_')[1].split('.')[0])
                self.data_list.append({
                    "path": os.path.join(img_dir, img_name),
                    "label": label
                })

    def _load_aptos(self):
        csv_path = os.path.join(self.data_dir, "train.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.data_list.append({
                "path": os.path.join(self.data_dir, "train_images", f"{row['id_code']}.png"),
                "label": int(row['diagnosis'])
            })

    def _load_eyepacs(self):
        csv_path = os.path.join(self.data_dir, "trainLabels.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.data_list.append({
                "path": os.path.join(self.data_dir, "train", f"{row['image']}.jpeg"),
                "label": int(row['level'])
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['path']).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(item['label'], dtype=torch.long),
            "session_id": self.session_id
        }

def get_incremental_loader(session_id, data_root, batch_size=16, shot=10):
    """
    특정 세션의 데이터로더를 반환하는 함수.
    Incremental Session은 소량의 데이터로 학습하므로 작은 batch_size를 권장합니다.
    """
    dataset = UnifiedIncrementalDataset(
        session_id=session_id, 
        data_dir=data_root, 
        shot=shot
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)