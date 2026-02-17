"""
FGADR 데이터셋 구조 진단 스크립트
서버에서 실행: python diagnose_fgadr.py
"""
import os
import glob

FGADR_ROOT = '/root/DICAN_DATASETS/FGADR'

print("=" * 60)
print("FGADR Dataset Structure Diagnosis")
print("=" * 60)

seg_root = os.path.join(FGADR_ROOT, "Seg-set")
if not os.path.exists(seg_root):
    print(f"[ERROR] {seg_root} not found!")
    exit()

print(f"\n[1] Seg-set 하위 폴더:")
for item in sorted(os.listdir(seg_root)):
    full = os.path.join(seg_root, item)
    if os.path.isdir(full):
        count = len(os.listdir(full))
        samples = sorted(os.listdir(full))[:3]
        print(f"  📁 {item}/ → {count} files")
        print(f"     예시: {samples}")
    else:
        print(f"  📄 {item}")

csv_path = os.path.join(seg_root, "DR_Seg_Grading_Label.csv")
if os.path.exists(csv_path):
    print(f"\n[2] CSV 파일 분석:")
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    print(f"  총 행 수: {len(lines)} (헤더 포함)")
    print(f"  처음 5행:")
    for i, line in enumerate(lines[:6]):
        print(f"    [{i}] {line.strip()}")
    
    header = lines[0].strip()
    sep = ',' if ',' in header else '\t' if '\t' in header else ' '
    cols = header.split(sep)
    print(f"\n  구분자: '{sep}', 컬럼: {cols}")
    
    if len(lines) > 1:
        from collections import Counter
        labels = []
        for line in lines[1:]:
            parts = line.strip().split(sep)
            if len(parts) >= 2:
                try: labels.append(int(parts[-1]))
                except: pass
        if labels:
            dist = Counter(labels)
            print(f"\n  라벨 분포:")
            for k in sorted(dist.keys()):
                print(f"    Grade {k}: {dist[k]}개")
            print(f"    총: {sum(dist.values())}개")

img_dir = os.path.join(seg_root, "Original_Images")
mask_dirs = {
    "EX": os.path.join(seg_root, "HardExudate_Masks"),
    "HE": os.path.join(seg_root, "Hemohedge_Masks"),
    "MA": os.path.join(seg_root, "Microaneurysms_Masks"),
    "SE": os.path.join(seg_root, "SoftExudate_Masks"),
}

print(f"\n[3] 이미지-마스크 매칭:")
if os.path.exists(img_dir):
    imgs = sorted(os.listdir(img_dir))
    print(f"  Original_Images: {len(imgs)} files")
    print(f"  예시: {imgs[:5]}")
    if imgs:
        sample_id = os.path.splitext(imgs[0])[0]
        sample_ext = os.path.splitext(imgs[0])[1]
        print(f"  샘플 ID: {sample_id}, ext: {sample_ext}")
        for concept, md in mask_dirs.items():
            if os.path.exists(md):
                mf = set(os.listdir(md))
                found = None
                for ext in ['.png', '.bmp', '.tif', '.jpg', sample_ext]:
                    if sample_id + ext in mf:
                        found = sample_id + ext
                        break
                if found:
                    print(f"    {concept}: ✅ {found} (총 {len(mf)}개)")
                else:
                    print(f"    {concept}: ❌ 불일치, 마스크 예시: {sorted(list(mf))[:3]}")

print(f"\n[4] 마스크 확장자:")
for concept, md in mask_dirs.items():
    if os.path.exists(md):
        files = os.listdir(md)
        exts = set(os.path.splitext(f)[1].lower() for f in files)
        print(f"  {concept}: {exts}, {len(files)}개")

print(f"\n[5] 마스크 픽셀값:")
try:
    from PIL import Image
    import numpy as np
    for concept, md in mask_dirs.items():
        if os.path.exists(md):
            files = sorted(os.listdir(md))
            non_empty = 0
            for f in files[:20]:
                m = np.array(Image.open(os.path.join(md, f)).convert("L"))
                if m.max() > 0: non_empty += 1
            print(f"  {concept}: {non_empty}/20 non-empty, shape={m.shape}, range=[{m.min()},{m.max()}]")
except:
    print("  PIL unavailable")

print("\n" + "=" * 60)
print("진단 완료. 위 결과를 붙여넣어 주세요.")