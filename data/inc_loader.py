import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class UnifiedIncrementalDataset(Dataset):
    """
    [Incremental Session용 통합 데이터셋 - 최종 수정 (File-based)]
    - 모든 데이터셋(APTOS, Messidor-2, DRAC22)을 압축 해제된 폴더에서 로드합니다.
    - ZIP 관련 복잡한 로직을 제거하고 'mode="file"'로 통일했습니다.
    - APTOS는 train.csv만 존재하므로, 이를 Train/Val/Test로 내부 분할합니다.
    """
    def __init__(self, session_id, data_dir, img_size=224, shot=10, split='train', seed=42):
        """
        Args:
            session_id (int): 1(APTOS), 2(Messidor-2), 3(DRAC22)
            data_dir (str): 'DICAN_DATASETS' 폴더 경로
            split (str): 'train', 'val', 'test'
        """
        self.session_id = session_id
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.seed = seed
        self.data_list = []
        
        # 1. 세션별 데이터 로드
        if session_id == 1: # APTOS 2019
            self._load_aptos()
        elif session_id == 2: # Messidor-2
            self._load_messidor2()
        elif session_id == 3: # DRAC22
            self._load_drac22()
        else:
            raise ValueError(f"Unknown session_id: {session_id}")
        
        if session_id == 3:
            self._map_drac_labels()
            
        # 2. Few-shot 샘플링 (Train set에만 적용)
        # Test/Val 셋은 전체 데이터를 사용하여 공정한 평가를 함
        if split == 'train' and shot is not None:
            self._apply_few_shot(shot)
        
        # 3. 전처리
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _apply_few_shot(self, shot):
        """클래스별 균등 샘플링 (K-shot)"""
        df = pd.DataFrame(self.data_list)
        few_shot_data = []
        
        for grade in range(5):
            subset = df[df['label'] == grade]
            if len(subset) > 0:
                n_samples = min(len(subset), shot)
                sampled = subset.sample(n=n_samples, random_state=self.seed)
                few_shot_data.extend(sampled.to_dict('records'))
                
        self.data_list = few_shot_data
        print(f"[{self.split.upper()}] Session {self.session_id}: Few-shot applied -> {len(self.data_list)} samples.")

    def _split_data(self, all_data):
        """
        전체 데이터 리스트를 Train(80%) / Val(10%) / Test(10%)로 분할
        """
        # Stratified Split을 위해 라벨 추출
        labels = [x['label'] for x in all_data]

        # 1. Train / Temp 분할 (8:2)
        train_data, temp_data = train_test_split(
            all_data, test_size=0.2, random_state=self.seed, stratify=labels
        )
        
        # Temp 데이터의 라벨 추출
        temp_labels = [x['label'] for x in temp_data]

        # 2. Temp를 Val / Test 분할 (1:1 -> 전체의 10%:10%)
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=self.seed, stratify=temp_labels
        )
        
        if self.split == 'train':
            return train_data
        elif self.split == 'val':
            return val_data
        else: # test
            return test_data

    # =========================================================
    # 데이터셋별 로딩 로직 (폴더 구조 반영)
    # =========================================================

    def _load_aptos(self):
        """
        [Structure Check]
        aptos/
          aptos2019-blindness-detection/
            train_images/
            train.csv
        """
        root = os.path.join(self.data_dir, "aptos", "aptos2019-blindness-detection")
        csv_path = os.path.join(root, "train.csv")
        img_dir = os.path.join(root, "train_images")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"APTOS CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        all_data = []
        
        for _, row in df.iterrows():
            # id_code + .png
            img_name = f"{row['id_code']}.png"
            all_data.append({
                "path": os.path.join(img_dir, img_name),
                "label": int(row['diagnosis'])
            })
            
        # Train.csv 하나를 쪼개서 사용
        self.data_list = self._split_data(all_data)

    def _load_messidor2(self):
        root = os.path.join(self.data_dir, "messidor-2")
        img_dir = os.path.join(root, "IMAGES")
        csv_path = os.path.join(root, "messidor_data.csv")

        if not os.path.exists(csv_path):
            # 혹시 CSV 이름이 다를 수 있으니 예비책
            csv_path = os.path.join(root, "messidor_2.csv") 

        df = pd.read_csv(csv_path)
        
        # 라벨 없는 행 제거
        original_len = len(df)
        df = df.dropna(subset=['adjudicated_dr_grade'])
        if len(df) < original_len:
                print(f"  [Messidor-2] Dropped {original_len - len(df)} rows with missing labels.")

        all_data = []
        for _, row in df.iterrows():
            raw_id = str(row['image_id'])
            
            # [핵심 수정] 확장자가 붙어있으면 일단 떼어냄 (중복 방지)
            if raw_id.lower().endswith('.jpg'):
                base_id = raw_id[:-4]
            else:
                base_id = raw_id
            
            # 후보군 생성: 소문자, 대문자, 원본
            candidates = [base_id + ".jpg", base_id + ".JPG", raw_id]
            
            found_path = None
            for cand in candidates:
                full_path = os.path.join(img_dir, cand)
                if os.path.exists(full_path):
                    found_path = full_path
                    break
            
            if found_path:
                all_data.append({
                    "path": found_path,
                    "label": int(float(row['adjudicated_dr_grade']))
                })
            else:
                # 못 찾았을 때만 조용히 로그 (너무 많이 뜨면 주석 처리)
                # print(f"[Warning] Image not found: {base_id}")
                pass

        self.data_list = self._split_data(all_data)

    def _load_drac22(self):
        """
        [Structure Check]
        DRAC22/
          C. Diabetic Retinopathy Grading/
            1. Original Images/
              a. Training Set/
            2. Groundtruths/
              a. DRAC2022_ ... Labels.csv
        """
        root = os.path.join(self.data_dir, "DRAC22", "C. Diabetic Retinopathy Grading")
        gt_folder = os.path.join(root, "2. Groundtruths")
        img_root = os.path.join(root, "1. Original Images", "a. Training Set")

        # CSV 파일명 찾기
        try:
            csv_name = [f for f in os.listdir(gt_folder) if "Training Labels" in f and f.endswith(".csv")][0]
            csv_path = os.path.join(gt_folder, csv_name)
        except IndexError:
            raise FileNotFoundError(f"DRAC22 Label CSV not found in {gt_folder}")
        
        df = pd.read_csv(csv_path)
        all_data = []
        
        for _, row in df.iterrows():
            # [수정 3] Warning 해결: row[0] 대신 row.iloc[0] 사용
            img_name = row.iloc[0] 
            grade = int(row.iloc[1])
            
            all_data.append({
                "path": os.path.join(img_root, img_name),
                "label": grade
            })
            
        self.data_list = self._split_data(all_data)

    def _map_drac_labels(self):
        """
        [DRAC22 Label Mapping]
        Original: 0(Normal), 1(NPDR), 2(PDR)
        Target  : 0(Normal), 2(Moderate), 4(Proliferative)
        """
        print(f"[*] Remapping DRAC22 Labels for Task 3...")
        
        mapped_count = 0
        for i in range(len(self.data_list)):
            item = self.data_list[i]
            original_label = int(item['label'])
            
            # 매핑 로직
            if original_label == 0:
                new_label = 0
            elif original_label == 1:
                new_label = 2  # NPDR -> Moderate (2)
            elif original_label == 2:
                new_label = 4  # PDR -> Proliferative (4)
            else:
                new_label = original_label
            
            # 변경 적용
            if original_label != new_label:
                self.data_list[i]['label'] = new_label
                mapped_count += 1
                
        print(f"    -> {mapped_count} labels remapped to Global Standard (0, 2, 4).")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        try:
            # 모든 데이터셋이 이제 일반 파일 경로를 사용
            image = Image.open(item['path']).convert("RGB")
        except Exception as e:
            print(f"Error loading image {item['path']}: {e}")
            # 에러 발생 시 검은색 이미지 반환 (학습 중단 방지)
            image = Image.new('RGB', (self.img_size, self.img_size))

        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": torch.tensor(item['label'], dtype=torch.long),
            "session_id": self.session_id
        }

def get_incremental_loader(session_id, data_root, mode='train', batch_size=16, shot=10):
    """
    Args:
        mode (str): 'train', 'val', 'test'
    """
    # Test/Val 모드일 때는 Shot 제한 없이 전체 데이터 사용
    current_shot = shot if mode == 'train' else None
    
    dataset = UnifiedIncrementalDataset(
        session_id=session_id, 
        data_dir=data_root, 
        shot=current_shot,
        split=mode
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode == 'train'), 
        num_workers=4,
        pin_memory=True
    )

if __name__ == "__main__":
    import sys
    
    # -------------------------------------------------------------------------
    # [사용자 설정] 실제 데이터셋이 있는 루트 폴더 경로로 수정해주세요.
    # 예: '/Users/nyoung/DICAN_DATASETS'
    DATA_ROOT = '/root/DICAN_DATASETS' 
    # -------------------------------------------------------------------------

    print(f"[*] Testing Incremental Loader with DATA_ROOT: {DATA_ROOT}\n")
    
    if not os.path.exists(DATA_ROOT):
        print(f"[Error] Directory not found: {DATA_ROOT}")
        print("Please update 'DATA_ROOT' variable in the __main__ block.")
        sys.exit(1)

    # 테스트할 세션 ID 목록 (1: APTOS, 2: Messidor-2, 3: DRAC22)
    session_map = {1: "APTOS 2019", 2: "Messidor-2", 3: "DRAC22"}
    
    for sid, name in session_map.items():
        print(f"="*60)
        print(f"Start Checking Session {sid}: {name}")
        print(f"="*60)
        
        try:
            # ----------------------------------------------------------------
            # 1. Train Loader 테스트 (Few-shot 적용 확인)
            # ----------------------------------------------------------------
            SHOT_NUM = 5  # 테스트용으로 5-shot 설정 (총 데이터 약 25개 예상)
            BATCH_SIZE = 4
            
            train_loader = get_incremental_loader(
                session_id=sid, 
                data_root=DATA_ROOT, 
                mode='train', 
                batch_size=BATCH_SIZE, 
                shot=SHOT_NUM
            )
            
            train_size = len(train_loader.dataset)
            print(f"  [Train] Loader created.")
            print(f"    -> Dataset size: {train_size} images")
            print(f"    -> Expected size: <= {5 * SHOT_NUM} (5 classes * {SHOT_NUM} shot)")
            
            # 배치 로딩 시도 (이미지 파일 읽기 테스트)
            batch = next(iter(train_loader))
            images = batch['image']
            labels = batch['label']
            
            print(f"    -> Batch Fetch Success!")
            print(f"    -> Image Shape: {images.shape} (Batch, C, H, W)")
            print(f"    -> Label Shape: {labels.shape}")
            print(f"    -> Sample Labels: {labels.tolist()}")
            
            # Few-shot 검증
            if train_size <= 5 * SHOT_NUM:
                print("    -> [PASS] Few-shot sampling seems correct.")
            else:
                print("    -> [WARNING] Dataset size is larger than expected for few-shot.")

            # ----------------------------------------------------------------
            # 2. Validation Loader 테스트 (Split 및 전체 데이터 로드 확인)
            # ----------------------------------------------------------------
            val_loader = get_incremental_loader(
                session_id=sid, 
                data_root=DATA_ROOT, 
                mode='val', 
                batch_size=BATCH_SIZE
            )
            
            val_size = len(val_loader.dataset)
            print(f"\n  [Val] Loader created (Split Check).")
            print(f"    -> Dataset size: {val_size} images")
            
            # Val 데이터셋은 Few-shot이 적용되지 않으므로 Train보다 훨씬 커야 함
            if val_size > train_size:
                print(f"    -> [PASS] Validation set is larger than Few-shot Train set.")
            else:
                print(f"    -> [WARNING] Validation set is unusually small. Check split logic.")
                
            print(f"\n  [SUCCESS] Session {sid} ({name}) is ready for training!\n")

        except FileNotFoundError as e:
            print(f"\n  [FAIL] File missing in Session {sid}: {e}")
            print("  Please check your folder structure and file names.")
        except Exception as e:
            print(f"\n  [FAIL] Error in Session {sid}: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n")