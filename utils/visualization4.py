import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 1. 데이터 준비
concepts = ['EX', 'HE', 'MA', 'SE']

# Before: 직교화 적용 전 (Prototype Similarity Matrix)
matrix_before = np.array([
    [1.000, 0.778, 0.952, 0.963],
    [0.778, 1.000, 0.853, 0.872],
    [0.952, 0.853, 1.000, 0.987],
    [0.963, 0.872, 0.987, 1.000]
])

# After: 직교화 적용 후 (Prototype Centroid Similarity Matrix)
matrix_after = np.array([
    [1.000, 0.349, 0.338, 0.319],
    [0.349, 1.000, 0.290, 0.361],
    [0.338, 0.290, 1.000, 0.141],
    [0.319, 0.361, 0.141, 1.000]
])

# 2. 그래프 그리기 (1행 2열)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# [핵심 수정] 파랑(낮음) -> 하양(중간) -> 빨강(높음) 테마 적용
cmap = 'coolwarm' 

# --- 첫 번째 히트맵: Before ---
sns.heatmap(matrix_before, ax=axes[0], annot=True, fmt=".3f", cmap=cmap,
            xticklabels=concepts, yticklabels=concepts, 
            vmin=0, vmax=1, # 범위를 0~1로 고정하여 두 그래프의 색상 기준 통일
            annot_kws={"size": 13, "weight": "bold"}, cbar_kws={"shrink": 0.8})
axes[0].set_title('Before: Prototype Similarity\n(Concept Entanglement)', 
                  fontsize=15, fontweight='bold', pad=15)
axes[0].tick_params(axis='both', labelsize=12)

# --- 두 번째 히트맵: After ---
sns.heatmap(matrix_after, ax=axes[1], annot=True, fmt=".3f", cmap=cmap,
            xticklabels=concepts, yticklabels=concepts, 
            vmin=0, vmax=1, 
            annot_kws={"size": 13, "weight": "bold"}, cbar_kws={"shrink": 0.8})
axes[1].set_title('After: Prototype Centroid Similarity\n(Soft Orthogonalization Applied)', 
                  fontsize=15, fontweight='bold', pad=15)
axes[1].tick_params(axis='both', labelsize=12)

# 전체 타이틀 
fig.suptitle('Ablation Study: Effect of Gram-Schmidt Soft Orthogonalization', 
             fontsize=18, fontweight='bold', y=1.05)

plt.tight_layout()
plt.savefig('orthogonalization_ablation_coolwarm.png', dpi=300, bbox_inches='tight')
plt.show()