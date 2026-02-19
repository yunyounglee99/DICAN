import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 실험 데이터
data = {
    'Model': ['LwF', 'EWC', 'L2P', 'DualPrompt', 'DICAN (Ours)'],
    'Avg Acc': [11.94, 24.39, 44.37, 41.96, 54.37],
    'BWT': [-38.49, -33.94, -8.93, -16.42, -8.45],
    'FWT': [2.70, 0.00, 0.00, 0.00, 49.39],
    'Forgetting': [40.78, 40.23, 19.55, 27.41, 9.59],
    'Avg QWK': [0.08, 0.06, 0.31, 0.27, 0.48]
}

df = pd.DataFrame(data)

# 2. 그래프 그리기
fig, axes = plt.subplots(1, 5, figsize=(16, 5))

metrics = ['Avg Acc', 'BWT', 'FWT', 'Forgetting', 'Avg QWK']

# [핵심 수정] CL 논문에 적합한 LaTeX 수식 제목 적용
titles = [
    r'Final $A_N$ (%)',         # 마지막 세션 평균 정확도
    r'Final BWT (%)',           # 마지막 세션 Backward Transfer
    r'Final FWT (%)',           # 마지막 세션 Forward Transfer
    r'Final Forgetting (%)',    # 마지막 세션 Forgetting
    r'Final $QWK_N$'            # 마지막 세션 QWK
]

colors = ['#BDC3C7', '#BDC3C7', '#BDC3C7', '#BDC3C7', '#FF7F50'] 

for i, metric in enumerate(metrics):
    bars = axes[i].bar(df['Model'], df[metric], color=colors, edgecolor='black', linewidth=0.8)
    # 제목 설정
    axes[i].set_title(titles[i], fontweight='bold', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Y축 범위 설정
    y_min, y_max = df[metric].min(), df[metric].max()
    if y_min >= 0:
        axes[i].set_ylim(0, max(y_max * 1.2, 0.1)) # 위쪽 여유 20%
        axes[i].axhline(0, color='black', linewidth=1)
    else:
        y_range = y_max - y_min
        axes[i].set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.2)
        axes[i].axhline(0, color='black', linewidth=0.8)

    # 텍스트 위치 및 정렬(Alignment) 로직
    current_ylim = axes[i].get_ylim()
    ylim_range = current_ylim[1] - current_ylim[0]
    
    for bar in bars:
        height = bar.get_height()
        
        # QWK 등 소수점 포맷
        fmt = '{:.2f}' if (metric == 'Avg QWK' or abs(height) < 1.0) else '{:.1f}'
        
        # 정렬 기준(va)을 변경하여 겹침 원천 차단
        if height >= 0:
            offset = ylim_range * 0.02 
            va_align = 'bottom'
        else:
            offset = -ylim_range * 0.02
            va_align = 'top'

        axes[i].text(bar.get_x() + bar.get_width()/2., height + offset,
                     fmt.format(height), 
                     ha='center', va=va_align,
                     fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('dican_comparison_plot_final.png', dpi=300)
plt.show()