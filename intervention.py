"""
DICAN Concept Intervention Analysis
=====================================
CBM(Concept Bottleneck Model)의 핵심 속성인 **Intervenability**를 검증.

[목적]
의사가 모델의 중간 Concept Score를 수동으로 수정(Intervention)했을 때,
최종 DR Grade 예측이 의학적으로 논리적인 방향으로 변하는지 확인.

[CBM Intervenability 정의]
"중간 개념 C의 값을 사람이 수동으로 수정하면,
 최종 예측 Y도 그에 맞게 논리적으로 변해야 한다."

[실험 시나리오]
1. Single Concept Intervention:
   하나의 Concept Score를 강제로 높이거나 낮추고 예측 변화 관찰
   예: "출혈(HE) score를 max로 올리면 Grade가 올라가는가?"

2. Multi-Concept Intervention:
   여러 Concept을 동시에 변경
   예: "모든 병변을 0으로 → Grade 0이 되는가?"

3. Grade-Guided Intervention:
   특정 Grade에 해당하는 의학적 규칙에 맞게 Concept 설정
   예: "Grade 3 규칙(EX=1, HE=1, MA=1, SE=1) 적용 → Grade 3 예측?"

[사용법]
  # 기본 (단일 샘플 분석, checkpoint 필요)
  python intervention.py \\
      --checkpoint_dir ./checkpoints \\
      --data_path /root/DICAN_DATASETS \\
      --task_id 1 \\
      --sample_idx 0

  # 특정 Concept 개입 (EX를 max로)
  python intervention.py \\
      --checkpoint_dir ./checkpoints \\
      --data_path /root/DICAN_DATASETS \\
      --task_id 0 \\
      --sample_idx 5 \\
      --intervene_concept EX \\
      --intervene_value 15.0

  # 전체 분석 (모든 Concept에 대해 sweep)
  python intervention.py \\
      --checkpoint_dir ./checkpoints \\
      --data_path /root/DICAN_DATASETS \\
      --task_id 1 \\
      --sample_idx 0 \\
      --full_sweep
      
  # 결과를 파일로 저장
  python intervention.py \\
      --checkpoint_dir ./checkpoints \\
      --data_path /root/DICAN_DATASETS \\
      --full_sweep \\
      --save_results ./results/intervention_analysis.pt
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from models import DICAN_CBM


# =================================================================
# Concept 상수 정의
# =================================================================
CONCEPT_NAMES = ["EX", "HE", "MA", "SE"]
GRADE_NAMES = ["Normal (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]

# Score 구조: [EX_max, HE_max, MA_max, SE_max, EX_mean, ..., EX_std, ...]
# 총 12차원 (4 concepts × 3 statistics)
NUM_CONCEPTS = 4

# 의학적 규칙: Grade별 병변 존재 기대치 (README.md 기반)
MEDICAL_RULES = {
    0: {"EX": 0, "HE": 0, "MA": 0, "SE": 0},  # Normal
    1: {"EX": 0, "HE": 0, "MA": 1, "SE": 0},  # Mild: MA only
    2: {"EX": 1, "HE": 1, "MA": 1, "SE": 0},  # Moderate: EX, HE, MA
    3: {"EX": 1, "HE": 1, "MA": 1, "SE": 1},  # Severe: All
    4: {"EX": 1, "HE": 1, "MA": 1, "SE": 1},  # Proliferative: All
}

CONCEPT_TO_IDX = {name: i for i, name in enumerate(CONCEPT_NAMES)}


# =================================================================
# 모델 로딩
# =================================================================
def load_model(checkpoint_dir, device, projector_type='lora', 
               projector_kwargs=None, num_clusters=3):
    """
    체크포인트에서 모델 로드.
    
    로드 우선순위:
      1. phase1c_best.pth (Phase 1-A/B/C 완료)
      2. best_base_model.pth
      3. phase1a_best.pth
    + base_prototypes.pt (Prototype Bank)
    """
    if projector_kwargs is None:
        projector_kwargs = {}
    
    model = DICAN_CBM(
        num_concepts=NUM_CONCEPTS,
        num_classes=5,
        feature_dim=2048,
        num_clusters=num_clusters,
        projector_type=projector_type,
        projector_kwargs=projector_kwargs
    ).to(device)
    
    # 모델 가중치 로드
    model_candidates = [
        "phase1c_best.pth",
        "best_base_model.pth",
        "phase1a_best.pth",
    ]
    
    loaded = False
    for fname in model_candidates:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            print(f"[✓] Model loaded: {path}")
            loaded = True
            break
    
    if not loaded:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            f"Expected: {model_candidates}"
        )
    
    # Prototype 로드
    proto_path = os.path.join(checkpoint_dir, "base_prototypes.pt")
    if os.path.exists(proto_path):
        learned_scale = model.prototypes.logit_scale.data.clone()
        model.prototypes.load_prototypes(proto_path)
        model.prototypes.logit_scale.data = learned_scale
        print(f"[✓] Prototypes loaded: {proto_path}")
    else:
        print(f"[⚠️] Prototypes not found: {proto_path}")
    
    return model


def load_inc_checkpoint(model, checkpoint_dir, task_id, device):
    """Incremental Task 체크포인트 로드 (있으면)"""
    path = os.path.join(checkpoint_dir, f"inc_task{task_id}_full.pth")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        acc = ckpt.get('accuracy', 0.0)
        print(f"[✓] Inc Task {task_id} loaded: {path} (Acc: {acc:.2f}%)")
        return True
    return False


# =================================================================
# 데이터 로딩 (단일 샘플)
# =================================================================
def get_sample(data_path, task_id, sample_idx, split='test'):
    """
    특정 Task의 특정 샘플을 가져옴.
    
    task_id=0: Base (DDR val set)
    task_id=1~3: Incremental (APTOS, Messidor-2, DRAC22)
    """
    if task_id == 0:
        from data.base_loader import DDRBaseDataset
        dataset = DDRBaseDataset(
            root_dir=os.path.join(data_path, 'DDR'), 
            split='valid'
        )
    else:
        from data.inc_loader import UnifiedIncrementalDataset
        dataset = UnifiedIncrementalDataset(
            session_id=task_id,
            data_dir=data_path,
            split=split,
            shot=None  # 전체 데이터 사용
        )
    
    if sample_idx >= len(dataset):
        print(f"[Warning] sample_idx={sample_idx} > dataset size={len(dataset)}, using 0")
        sample_idx = 0
    
    sample = dataset[sample_idx]
    return sample, dataset


# =================================================================
# Intervention 핵심 로직
# =================================================================
class ConceptIntervenor:
    """
    CBM Concept Intervention 분석기.
    
    [Score 구조 해설]
    concept_scores: [1, 12] 텐서
    
    인덱스 매핑:
      [0:4]   = Max scores  (EX_max, HE_max, MA_max, SE_max)
      [4:8]   = Mean scores (EX_mean, HE_mean, MA_mean, SE_mean)  
      [8:12]  = Std scores  (EX_std, HE_std, MA_std, SE_std)
    
    Intervention 시 Max와 Mean을 동시에 변경하고,
    Std는 0으로 설정 (확실한 존재/부재를 의미).
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.nc = NUM_CONCEPTS  # 4
    
    def get_original_prediction(self, image):
        """
        원본 이미지에 대한 예측 수행.
        
        Returns:
            dict: {
                'concept_scores': [1, 12] 원본 Concept Score,
                'logits': [1, 5] 원본 Logit,
                'probs': [1, 5] Softmax 확률,
                'pred_grade': int 예측 등급,
                'features': [1, 2048, 7, 7] 특징 맵
            }
        """
        self.model.set_session_mode('intervention')
        
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0).to(self.device))
        
        concept_scores = outputs['concept_scores']  # [1, 12]
        logits = self.model.predict_from_concepts(concept_scores)
        probs = F.softmax(logits, dim=1)
        pred_grade = logits.argmax(dim=1).item()
        
        return {
            'concept_scores': concept_scores.cpu(),
            'logits': logits.cpu(),
            'probs': probs.cpu(),
            'pred_grade': pred_grade,
            'features': outputs['features'].cpu(),
            'spatial_sim_map': outputs['spatial_sim_map'].cpu() 
                if outputs['spatial_sim_map'] is not None else None
        }
    
    def intervene_single_concept(self, original_scores, concept_name, value, 
                                  also_set_mean=True, zero_std=True):
        """
        단일 Concept에 개입.
        
        Args:
            original_scores: [1, 12] 원본 scores
            concept_name: 'EX', 'HE', 'MA', 'SE'
            value: 설정할 Max score 값
            also_set_mean: True면 Mean도 value의 50%로 설정
            zero_std: True면 Std를 0으로 (확실한 존재/부재 의미)
        
        Returns:
            intervened_scores: [1, 12] 수정된 scores
        """
        idx = CONCEPT_TO_IDX[concept_name]
        modified = original_scores.clone()
        
        # Max score 설정
        modified[0, idx] = value
        
        # Mean score 설정 (Max의 50%)
        if also_set_mean:
            modified[0, self.nc + idx] = value * 0.5
        
        # Std를 0으로 (확정적)
        if zero_std:
            modified[0, 2 * self.nc + idx] = 0.0
        
        return modified
    
    def intervene_by_medical_rule(self, original_scores, target_grade, 
                                   high_value=15.0, low_value=-5.0):
        """
        의학적 규칙에 따라 모든 Concept을 설정.
        
        Args:
            original_scores: [1, 12] 원본
            target_grade: 0~4 (설정할 Grade)
            high_value: 병변 존재 시 설정할 값
            low_value: 병변 부재 시 설정할 값
        
        Returns:
            intervened_scores: [1, 12]
        """
        rule = MEDICAL_RULES[target_grade]
        modified = original_scores.clone()
        
        for concept_name, expected in rule.items():
            idx = CONCEPT_TO_IDX[concept_name]
            val = high_value if expected == 1 else low_value
            
            modified[0, idx] = val                  # Max
            modified[0, self.nc + idx] = val * 0.5  # Mean
            modified[0, 2 * self.nc + idx] = 0.0    # Std
        
        return modified
    
    def intervene_suppress_all(self, original_scores, suppress_value=-5.0):
        """모든 Concept을 억제 (정상으로 만들기)"""
        return self.intervene_by_medical_rule(
            original_scores, target_grade=0, 
            high_value=suppress_value, low_value=suppress_value
        )
    
    def predict_from_scores(self, concept_scores):
        """수정된 Concept Scores로 예측 수행"""
        scores_device = concept_scores.to(self.device)
        with torch.no_grad():
            logits = self.model.predict_from_concepts(scores_device)
        probs = F.softmax(logits, dim=1)
        pred_grade = logits.argmax(dim=1).item()
        return {
            'logits': logits.cpu(),
            'probs': probs.cpu(),
            'pred_grade': pred_grade
        }


# =================================================================
# 분석 함수들
# =================================================================
def print_scores_breakdown(scores, title="Concept Scores"):
    """Score 벡터를 보기 좋게 출력"""
    nc = NUM_CONCEPTS
    max_s = scores[0, :nc]
    mean_s = scores[0, nc:2*nc]
    std_s = scores[0, 2*nc:]
    
    print(f"\n  [{title}]")
    print(f"  {'Concept':>6} | {'Max':>8} | {'Mean':>8} | {'Std':>8}")
    print(f"  {'-'*40}")
    for i, name in enumerate(CONCEPT_NAMES):
        print(f"  {name:>6} | {max_s[i].item():>8.3f} | "
              f"{mean_s[i].item():>8.3f} | {std_s[i].item():>8.3f}")


def print_prediction(result, label=None, title="Prediction"):
    """예측 결과 출력"""
    probs = result['probs'][0]
    pred = result['pred_grade']
    
    print(f"\n  [{title}]")
    if label is not None:
        print(f"  Ground Truth: {GRADE_NAMES[label]}")
    print(f"  Predicted:    {GRADE_NAMES[pred]}")
    print(f"  Probabilities:")
    for g in range(5):
        bar = '█' * int(probs[g].item() * 30)
        marker = " ← PRED" if g == pred else ""
        gt_marker = " ← GT" if label is not None and g == label else ""
        print(f"    Grade {g}: {probs[g].item():>6.3f} {bar}{marker}{gt_marker}")


def run_single_concept_sweep(intervenor, original_scores, original_result, label=None):
    """
    각 Concept을 하나씩 높이거나 낮춰보고 예측 변화 관찰.
    
    CBM의 핵심 검증: "개념을 바꾸면 예측이 논리적으로 바뀌는가?"
    """
    print(f"\n{'='*60}")
    print(f"  Single Concept Intervention Sweep")
    print(f"  (각 Concept을 개별적으로 변경하고 예측 변화 관찰)")
    print(f"{'='*60}")
    
    original_pred = original_result['pred_grade']
    
    results = []
    
    # 각 Concept에 대해 높이기/낮추기
    test_values = [
        ("Suppress (→ -5.0)", -5.0),
        ("Mild (→ 3.0)", 3.0),
        ("Moderate (→ 8.0)", 8.0),
        ("Boost (→ 15.0)", 15.0),
    ]
    
    for concept_name in CONCEPT_NAMES:
        print(f"\n  ── Concept: {concept_name} ──")
        idx = CONCEPT_TO_IDX[concept_name]
        orig_val = original_scores[0, idx].item()
        print(f"  Original {concept_name}_max = {orig_val:.3f}")
        
        for desc, val in test_values:
            modified = intervenor.intervene_single_concept(
                original_scores, concept_name, val)
            result = intervenor.predict_from_scores(modified)
            
            new_pred = result['pred_grade']
            changed = "✅ CHANGED" if new_pred != original_pred else "  (same)"
            
            probs_str = " | ".join(f"G{g}={result['probs'][0][g]:.3f}" for g in range(5))
            
            print(f"    {desc}: Grade {original_pred} → {new_pred} {changed}")
            print(f"      [{probs_str}]")
            
            results.append({
                'concept': concept_name,
                'intervention': desc,
                'value': val,
                'original_pred': original_pred,
                'new_pred': new_pred,
                'probs': result['probs'].clone(),
                'changed': new_pred != original_pred
            })
    
    return results


def run_medical_rule_intervention(intervenor, original_scores, original_result, label=None):
    """
    의학적 규칙에 따라 모든 Concept을 설정하고 예측 확인.
    
    "Grade 3의 의학적 조건(EX=1, HE=1, MA=1, SE=1)으로 설정하면
     실제로 Grade 3을 예측하는가?"
    """
    print(f"\n{'='*60}")
    print(f"  Medical Rule-Based Intervention")
    print(f"  (의학적 규칙대로 Concept을 설정하고 예측 변화 관찰)")
    print(f"{'='*60}")
    
    original_pred = original_result['pred_grade']
    print(f"  Original prediction: {GRADE_NAMES[original_pred]}")
    if label is not None:
        print(f"  Ground truth:        {GRADE_NAMES[label]}")
    
    results = []
    
    for target_grade in range(5):
        rule = MEDICAL_RULES[target_grade]
        rule_str = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in rule.items())
        
        modified = intervenor.intervene_by_medical_rule(
            original_scores, target_grade)
        result = intervenor.predict_from_scores(modified)
        
        new_pred = result['pred_grade']
        match = "✅ MATCH" if new_pred == target_grade else f"⚠️ Got Grade {new_pred}"
        
        probs = result['probs'][0]
        top_prob = probs[new_pred].item()
        
        print(f"\n  Target Grade {target_grade} ({GRADE_NAMES[target_grade]}):")
        print(f"    Rule: {rule_str}")
        print(f"    Predicted: Grade {new_pred} (p={top_prob:.3f}) {match}")
        
        results.append({
            'target_grade': target_grade,
            'predicted_grade': new_pred,
            'matched': new_pred == target_grade,
            'probs': result['probs'].clone(),
            'confidence': top_prob
        })
    
    # 요약
    matches = sum(1 for r in results if r['matched'])
    print(f"\n  ── Summary ──")
    print(f"  Medical Rule Match Rate: {matches}/5 ({100*matches/5:.0f}%)")
    
    return results


def run_suppress_all_analysis(intervenor, original_scores, original_result, label=None):
    """
    "모든 병변을 억제하면 Grade 0이 되는가?" 검증
    
    CBM의 직관적 이해: 병변이 하나도 없으면 = 정상
    """
    print(f"\n{'='*60}")
    print(f"  Suppress-All Analysis")
    print(f"  (모든 Concept을 억제 → Grade 0이 되는가?)")
    print(f"{'='*60}")
    
    original_pred = original_result['pred_grade']
    
    modified = intervenor.intervene_suppress_all(original_scores)
    result = intervenor.predict_from_scores(modified)
    
    print_scores_breakdown(modified, "All Concepts Suppressed")
    print_prediction(result, label, "After Suppressing All")
    
    if result['pred_grade'] == 0:
        print(f"\n  ✅ Correct! Suppressing all concepts → Grade 0 (Normal)")
    else:
        print(f"\n  ⚠️ Expected Grade 0, got Grade {result['pred_grade']}")
    
    return result


def run_incremental_boost_analysis(intervenor, original_scores, original_result, label=None):
    """
    Concept을 하나씩 순차적으로 추가하며 Grade 변화 관찰.
    
    의학적 논리:
    1. 아무것도 없음 → Grade 0
    2. MA만 추가 → Grade 1 (Mild)
    3. MA + EX + HE → Grade 2 (Moderate)
    4. 모두 추가 → Grade 3~4 (Severe~Proliferative)
    """
    print(f"\n{'='*60}")
    print(f"  Incremental Concept Addition")
    print(f"  (병변을 하나씩 추가하며 Grade 상승 관찰)")
    print(f"{'='*60}")
    
    # 시작점: 모든 Concept 억제
    base_scores = intervenor.intervene_suppress_all(original_scores)
    result = intervenor.predict_from_scores(base_scores)
    print(f"\n  Step 0: All suppressed → Grade {result['pred_grade']}")
    
    additions = [
        ("MA only (Mild 기대)", ["MA"]),
        ("+ EX + HE (Moderate 기대)", ["MA", "EX", "HE"]),
        ("+ SE (Severe 기대)", ["MA", "EX", "HE", "SE"]),
    ]
    
    HIGH_VAL = 15.0
    LOW_VAL = -5.0
    
    results = [{'step': 0, 'concepts': [], 'pred': result['pred_grade']}]
    
    for step, (desc, active_concepts) in enumerate(additions, 1):
        modified = base_scores.clone()
        
        for cname in active_concepts:
            idx = CONCEPT_TO_IDX[cname]
            modified[0, idx] = HIGH_VAL            # Max
            modified[0, NUM_CONCEPTS + idx] = HIGH_VAL * 0.5  # Mean
            modified[0, 2 * NUM_CONCEPTS + idx] = 0.0         # Std
        
        result = intervenor.predict_from_scores(modified)
        pred = result['pred_grade']
        
        active_str = ", ".join(active_concepts)
        probs_str = " | ".join(f"G{g}={result['probs'][0][g]:.3f}" for g in range(5))
        
        print(f"  Step {step}: {desc}")
        print(f"    Active: [{active_str}] → Grade {pred}")
        print(f"    [{probs_str}]")
        
        results.append({
            'step': step,
            'concepts': active_concepts,
            'pred': pred,
            'probs': result['probs'].clone()
        })
    
    # Grade 단조 증가 확인
    grades = [r['pred'] for r in results]
    monotonic = all(grades[i] <= grades[i+1] for i in range(len(grades)-1))
    print(f"\n  Grade progression: {' → '.join(str(g) for g in grades)}")
    if monotonic:
        print(f"  ✅ Monotonically increasing (의학적으로 논리적)")
    else:
        print(f"  ⚠️ Non-monotonic (일부 비논리적 변화 존재)")
    
    return results


# =================================================================
# Main
# =================================================================
def main():
    parser = argparse.ArgumentParser(
        description='DICAN Concept Intervention Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 분석 (단일 샘플)
  python intervention.py --checkpoint_dir ./checkpoints --data_path /root/DICAN_DATASETS

  # 특정 Concept 개입
  python intervention.py --checkpoint_dir ./checkpoints --data_path /root/DICAN_DATASETS \\
      --intervene_concept HE --intervene_value 15.0

  # 전체 sweep 분석
  python intervention.py --checkpoint_dir ./checkpoints --data_path /root/DICAN_DATASETS --full_sweep

  # Incremental Task 1 (APTOS) 샘플 분석
  python intervention.py --checkpoint_dir ./checkpoints --data_path /root/DICAN_DATASETS \\
      --task_id 1 --sample_idx 0 --full_sweep
        """
    )
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='체크포인트 디렉토리')
    parser.add_argument('--data_path', type=str, default='/root/DICAN_DATASETS',
                        help='데이터셋 루트')
    parser.add_argument('--task_id', type=int, default=0,
                        help='분석할 Task ID (0=Base/DDR, 1=APTOS, 2=Messidor, 3=DRAC)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='분석할 샘플 인덱스')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_cluster', type=int, default=3)
    
    # Projector 설정 (체크포인트와 일치해야 함)
    parser.add_argument('--projector_type', type=str, default='lora',
                        choices=['lora', 'linear_1layer', 'linear_2layer'])
    parser.add_argument('--projector_rank', type=int, default=64)
    parser.add_argument('--projector_hidden', type=int, default=512)
    
    # Intervention 옵션
    parser.add_argument('--intervene_concept', type=str, default=None,
                        choices=['EX', 'HE', 'MA', 'SE'],
                        help='개입할 Concept (지정하지 않으면 원본만 분석)')
    parser.add_argument('--intervene_value', type=float, default=15.0,
                        help='Concept에 설정할 값')
    parser.add_argument('--full_sweep', action='store_true',
                        help='모든 분석 (single sweep + medical rule + suppress + incremental) 수행')
    parser.add_argument('--save_results', type=str, default=None,
                        help='결과를 .pt 파일로 저장')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # ─── 1. 모델 로드 ───
    print(f"\n{'='*60}")
    print(f"  DICAN Concept Intervention Analysis")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Task: {args.task_id}")
    print(f"  Sample: #{args.sample_idx}")
    print(f"  Projector: {args.projector_type}")
    
    projector_kwargs = {}
    if args.projector_type == 'lora':
        projector_kwargs['rank'] = args.projector_rank
    elif args.projector_type == 'linear_2layer':
        projector_kwargs['hidden_dim'] = args.projector_hidden
    
    model = load_model(
        args.checkpoint_dir, device, 
        projector_type=args.projector_type,
        projector_kwargs=projector_kwargs,
        num_clusters=args.num_cluster
    )
    
    # Incremental task면 해당 체크포인트도 로드
    if args.task_id > 0:
        loaded = load_inc_checkpoint(model, args.checkpoint_dir, args.task_id, device)
        if loaded:
            model.projector.mode = 'incremental'
        else:
            print(f"[Info] No inc checkpoint for task {args.task_id}, using base weights")
            model.projector.mode = 'base'
    else:
        model.projector.mode = 'base'
    
    # ─── 2. 샘플 로드 ───
    print(f"\n[*] Loading sample...")
    sample, dataset = get_sample(args.data_path, args.task_id, args.sample_idx)
    
    image = sample['image']
    label = sample['label']
    if isinstance(label, torch.Tensor):
        label = label.item()
    
    task_names = {0: "Base (DDR)", 1: "APTOS 2019", 2: "Messidor-2", 3: "DRAC22"}
    print(f"  Task: {task_names.get(args.task_id, f'Task {args.task_id}')}")
    print(f"  Ground Truth: {GRADE_NAMES[label]}")
    print(f"  Image shape: {image.shape}")
    
    # ─── 3. 원본 예측 ───
    intervenor = ConceptIntervenor(model, device)
    original = intervenor.get_original_prediction(image)
    
    print_scores_breakdown(original['concept_scores'], "Original Concept Scores")
    print_prediction(original, label, "Original Prediction")
    
    all_results = {
        'task_id': args.task_id,
        'sample_idx': args.sample_idx,
        'label': label,
        'original_scores': original['concept_scores'],
        'original_pred': original['pred_grade'],
        'original_probs': original['probs'],
    }
    
    # ─── 4. 특정 Concept 개입 ───
    if args.intervene_concept:
        print(f"\n{'='*60}")
        print(f"  Targeted Intervention: {args.intervene_concept} → {args.intervene_value}")
        print(f"{'='*60}")
        
        modified = intervenor.intervene_single_concept(
            original['concept_scores'], 
            args.intervene_concept, 
            args.intervene_value
        )
        
        print_scores_breakdown(modified, f"After Setting {args.intervene_concept}={args.intervene_value}")
        
        result = intervenor.predict_from_scores(modified)
        print_prediction(result, label, f"After Intervention ({args.intervene_concept}={args.intervene_value})")
        
        if result['pred_grade'] != original['pred_grade']:
            print(f"\n  ✅ Grade changed: {original['pred_grade']} → {result['pred_grade']}")
        else:
            print(f"\n  ── Grade unchanged: {original['pred_grade']}")
        
        all_results['targeted_intervention'] = {
            'concept': args.intervene_concept,
            'value': args.intervene_value,
            'modified_scores': modified,
            'new_pred': result['pred_grade'],
            'new_probs': result['probs']
        }
    
    # ─── 5. 전체 Sweep ───
    if args.full_sweep:
        sweep_results = run_single_concept_sweep(
            intervenor, original['concept_scores'], original, label)
        all_results['single_sweep'] = sweep_results
        
        medical_results = run_medical_rule_intervention(
            intervenor, original['concept_scores'], original, label)
        all_results['medical_rules'] = medical_results
        
        suppress_result = run_suppress_all_analysis(
            intervenor, original['concept_scores'], original, label)
        all_results['suppress_all'] = suppress_result
        
        incremental_results = run_incremental_boost_analysis(
            intervenor, original['concept_scores'], original, label)
        all_results['incremental_boost'] = incremental_results
    
    # ─── 6. 결과 저장 ───
    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(all_results, args.save_results)
        print(f"\n[✓] Results saved to: {args.save_results}")
    
    # ─── 7. 요약 ───
    print(f"\n{'='*60}")
    print(f"  Analysis Complete!")
    print(f"{'='*60}")
    print(f"  Sample: Task {args.task_id}, Index {args.sample_idx}")
    print(f"  GT: {GRADE_NAMES[label]} | Pred: {GRADE_NAMES[original['pred_grade']]}")
    
    if args.full_sweep:
        # Intervention 유효성 요약
        n_changed = sum(1 for r in sweep_results if r['changed'])
        n_total = len(sweep_results)
        med_matches = sum(1 for r in medical_results if r['matched'])
        
        print(f"\n  [Intervention Summary]")
        print(f"  Single Concept Sweep: {n_changed}/{n_total} interventions changed the prediction")
        print(f"  Medical Rule Match:   {med_matches}/5 grades matched")
        print(f"  Suppress-All → Grade 0: "
              f"{'✅ Yes' if suppress_result['pred_grade'] == 0 else '⚠️ No'}")


if __name__ == "__main__":
    main()