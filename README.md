# DICAN: Domain Invariant Concept Adjustment Network

**Mitigating Domain Shift in Concept Bottleneck Models for Diabetic Retinopathy Grading via Learnable Concept Projection**

> Yunyoung Lee (Korea University) · Eungjoo Lee (University of Arizona)
>
> *Submitted to SPIE Medical Imaging 2026*

---

## Overview

Concept Bottleneck Models (CBMs) offer interpretable diabetic retinopathy (DR) grading by routing predictions through clinically meaningful lesion concepts. However, they assume training and deployment data share the same distribution — an assumption that breaks down in practice, where fundus images vary substantially across hospital sites.

**DICAN** addresses this by freezing all diagnostic components (backbone, prototype bank, classification head) and training only a lightweight low-rank residual projector (262K parameters) to realign features when a new hospital domain arrives — with as few as **10 labeled images per grade**.

<img width="4240" height="2828" alt="dican drawio-2" src="https://github.com/user-attachments/assets/02a044b0-04e4-4773-b3b3-ad8a2f8e12c6" />


---

## Key Results

### Comparison with Baselines (10-shot FSDIL)

| Model | Avg. Accuracy (↑) | BWT (↑) | FWT (↑) | Forgetting (↓) | QWK (↑) |
|---|---|---|---|---|---|
| LwF | 11.94% | -38.49 | 2.70 | 40.78 | 0.08 |
| EWC | 24.39% | -33.94 | 0.00 | 40.23 | 0.06 |
| DualPrompt | 41.96% | -16.42 | 0.00 | 27.41 | 0.27 |
| L2P | 44.37% | -8.93 | 0.00 | 19.55 | 0.31 |
| **DICAN (Ours)** | **54.37%** | **-8.45** | **49.39** | **9.59** | **0.48** |

DICAN outperforms the strongest baseline (L2P) by **+10 percentage points** in accuracy while **halving forgetting**.

### Effect of Gram-Schmidt Orthogonalization (K=5)

| | Base | Inc 1 | Inc 2 | Inc 3 |
|---|---|---|---|---|
| W/ orthogonalization | 76.03% | 63.76% | 51.43% | 51.61% |
| W/o orthogonalization | 69.49% | 45.78% | 38.29% | 24.19% |

Without orthogonalization, inter-concept cosine similarity averages **0.95** → concept entanglement causes a **27-point accuracy collapse** by the final session.

### Projector Architecture Comparison (K=5, 10-shot)

| Projector | Params | Avg. Accuracy (↑) | Forgetting (↓) | QWK (↑) |
|---|---|---|---|---|
| Linear (1-layer) | 41.9M | 46.43% | 25.01 | 0.455 |
| Linear (2-layer) | 21.0M | 18.79% | 56.16 | 0.176 |
| **LoRA (Ours)** | **262K** | **57.43%** | **10.30** | **0.544** |

The LoRA projector uses **0.6% of the parameters** of the full-rank baseline while achieving the best accuracy and lowest forgetting.

---

## Method

### Three-Phase Base Session Learning

<img width="3204" height="3276" alt="DICAN_TRAINING_FINAL drawio" src="https://github.com/user-attachments/assets/db0cc492-c597-4196-91d7-4dd5630168f2" />

**Phase 1 — Segmentation-Guided Backbone Training**

A ResNet-50 backbone is trained with a U-Net decoder producing 4-channel pixel-level lesion predictions at 224×224. After training, the decoder is discarded; only backbone weights are retained.

```
Loss = Σ wᵢ (L_cls⁽ⁱ⁾ + λ_seg · L_seg⁽ⁱ⁾)
```
where `wᵢ = 1 / (N_classes · freq(cᵢ))` applies inverse class frequency weighting. Segmentation loss combines Dice (0.7) + focal BCE (0.3).

**Phase 2 — Multi-Cluster Prototype Extraction**

With the backbone frozen, K-Means++ (K=5 per concept) clusters mask-guided ROI features into a prototype bank capturing morphological diversity (e.g., dot vs. flame hemorrhages). Gram-Schmidt soft orthogonalization (β=0.5) separates inter-concept prototypes while preserving intra-concept diversity.

```
L_ortho = Σᵢ Σⱼ≠ᵢ ⟨c̄ᵢ, c̄ⱼ⟩² + λ_intra Σₖ Σₘ≠ₙ ⟨c̄ₖᵐ, c̄ₖⁿ⟩²
```

**Phase 3 — Concept Scoring and Head Training**

Best-match-then-aggregate scoring compresses sub-cluster similarities into 3 fixed statistics per concept:

| Statistic | Clinical Signal |
|---|---|
| **Max** | Lesion presence (strongest local activation) |
| **Mean** | Spatial extent / lesion severity |
| **Min** | Background floor (suppressed for normal retinas) |

This yields a fixed **12-dimensional bottleneck** (3 stats × 4 concepts) regardless of K — decoupling prototype granularity from head complexity.

### Incremental Session: Low-Rank Projector

When a new hospital arrives with only 10 labeled images per grade, only the projector is trained:

```
z' = z + W_up · σ(W_down · z)
```

where `W_down ∈ ℝ^(r×2048)`, `W_up ∈ ℝ^(2048×r)`, `r=64` (262K parameters total). Output layer initialized to zero for stable optimization.

**Incremental loss:**
```
L_inc = λ_ord · L_ordinal + λ_align · L_align + λ_sparsity · L_sparsity + λ_anc · ||θ - θ₀||²
```

---

## Datasets

| Session | Dataset | Images | Masks | Domain | Grades |
|---|---|---|---|---|---|
| Base | DDR | 6,835 | 275 | 147 hospitals, China | 5 |
| Base | FGADR | 1,566 | 1,566 | India | 5 |
| Inc 1 | APTOS 2019 | 3,662 | — | Aravind Eye Hosp., India | 5 |
| Inc 2 | Messidor-2 | 1,744 | — | Ophthalmology depts., France | 5 |
| Inc 3 | DRAC 2022 | 611 | — | Ultra-wide-field, China | 3 |

DRAC 2022 uses 3 coarse grades mapped to the 5-class ICDR scale (0→0, 1→2, 2→4), deliberately testing label granularity mismatch.

---

## Concept Intervention

A clinician can override concept statistics at inference time and observe deterministic changes in the diagnostic output — without retraining.

**Example:** A fundus image initially misclassified as Grade 2 (ground truth: Grade 3) due to an underestimated hemorrhage (HE) score of 7.70. Overriding HE Max to 15.0 corrects the prediction to Grade 3 with near-certain confidence.

```
Before intervention:  HE score = 7.70  →  Predicted: Grade 2 ✗
After intervention:   HE score = 15.0  →  Predicted: Grade 3 ✓
```

---

## Implementation Details

```
Backbone:        ResNet-50
GPU:             NVIDIA A100 (single)
Framework:       PyTorch

Phase 1:  AdamW (lr=1e-4, wd=5e-4), cosine annealing, 30 epochs, batch=32
Phase 2:  K-Means++ K=5, Gram-Schmidt β=0.5
Phase 3:  Adam (lr=1e-3, wd=1e-4), cosine annealing, 15 epochs
Inc:      Adam (lr=1e-3, wd=1e-4), 50 steps, λ_anchor=0.1
```

---

## Installation

```bash
git clone https://github.com/[your-repo]/DICAN.git
cd DICAN
pip install -r requirements.txt
```

---

## Usage

```bash
# Base session training
python train_base.py --dataset DDR FGADR --backbone resnet50

# Incremental session (new hospital domain)
python train_incremental.py --dataset APTOS --shot 10 --rank 64

# Evaluation
python evaluate.py --sessions all --metrics accuracy qwk forgetting
```

---

## Citation

```bibtex
@inproceedings{lee2026dican,
  title     = {Mitigating Domain Shift in Concept Bottleneck Models for Diabetic Retinopathy Grading via Learnable Concept Projection},
  author    = {Lee, Yunyoung and Lee, Eungjoo},
  booktitle = {SPIE Medical Imaging},
  year      = {2026}
}
```

---
