import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 실험 로그에서 추출된 Task별 평균 Acc 데이터
# (각 리스트는 Task 0 학습 후, Task 1 학습 후... 순서의 평균 정확도)
data = {
    'Session': ['Session 1', 'Session 2', 'Session 3', 'Session 4'],
    'DICAN (Ours)': [76.03, 63.76, 51.43, 51.61], # Session 3의 드롭은 로그 기반 수치임
    'L2P': [79.17, 57.59, 37.87, 44.37],
    'DualPrompt': [78.34, 52.28, 37.67, 41.96],
    'LwF': [70.06, 49.35, 19.58, 11.94],
    'EWC': [60.74, 46.75, 24.59, 24.39]
}

df = pd.DataFrame(data)

# 2. 그래프 시각화 설정
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# 모델별 선 그래프 그리기
models = ['LwF', 'EWC', 'L2P', 'DualPrompt', 'DICAN (Ours)']
colors = ['#95a5a6', '#7f8c8d', '#3498db', '#9b59b6', '#e74c3c'] # DICAN은 빨간색으로 강조
markers = ['s', '^', 'D', 'v', 'o'] # 각기 다른 마커 사용

for model, color, marker in zip(models, colors, markers):
    linewidth = 3 if 'DICAN' in model else 1.5
    plt.plot(df['Session'], df[model], label=model, color=color, 
             marker=marker, markersize=8, linewidth=linewidth, alpha=0.9)

# 3. 그래프 디테일 설정 (논문용)
plt.title('Task Accuracy Flow across Tasks', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Learning Sessions', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 90)
plt.grid(True, linestyle='--', alpha=0.6)

# 수치 강조 (최종 성능 지점)
final_session = df['Session'].iloc[-1]
plt.annotate(f"{df['DICAN (Ours)'].iloc[-1]:.1f}%", 
             xy=(final_session, df['DICAN (Ours)'].iloc[-1]),
             xytext=(10, 5), textcoords='offset points', 
             fontsize=12, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig('dican_accuracy_flow.png', dpi=300)
plt.show()