import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 실험 데이터 로드
data = {
    'Model': ['LwF', 'EWC', 'L2P', 'DualPrompt', 'DICAN (Ours)'],
    'Avg Acc': [11.94, 24.39, 44.37, 41.96, 49.64],
    'BWT': [-38.49, -33.94, -8.93, -16.42, 6.47],
    'FWT': [2.70, 0.00, 0.00, 0.00, 41.16],
    'Forgetting': [40.78, 40.23, 19.55, 27.41, 8.44],
    'Avg QWK': [0, 0, 0, 0, 0.219]
}

df = pd.DataFrame(data)

# 2. 그래프 그리기 (1행 5열)
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
metrics = ['Avg Acc', 'BWT', 'FWT', 'Forgetting', 'Avg QWK']
titles = ['Avg Accuracy (%)', 'BWT (%)', 'FWT (%)', 'Forgetting (%)', 'Avg QWK']
colors = ['#BDC3C7', '#BDC3C7', '#BDC3C7', '#BDC3C7', '#FF7F50'] # DICAN만 강조

for i, metric in enumerate(metrics):
    bars = axes[i].bar(df['Model'], df[metric], color=colors, edgecolor='black', linewidth=0.8)
    axes[i].set_title(titles[i], fontweight='bold', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45, labelsize=11)
    
    # 막대 상단에 값 표시
    for bar in bars:
        height = bar.get_height()
        offset = 1.0 if height >= 0 else -1.5
        axes[i].text(bar.get_x() + bar.get_width()/2., height + offset-0.7,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('dican_comparison_plot.png', dpi=300)
plt.show()