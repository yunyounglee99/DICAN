import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 준비
sessions = ['Session 1\n(Task 1)', 'Session 2\n(Task 1~2)', 'Session 3\n(Task 1~3)']
shots = ['5-Shot', '10-Shot', '15-Shot', '20-Shot']

# 각 샷 수에 따른 Session별 평균 정확도(A_N)
acc_5 = [68.28, 57.83, 41.52]
acc_10 = [71.92, 60.53, 41.89]
acc_15 = [73.04, 59.43, 40.88]
acc_20 = [73.30, 58.68, 40.05]

# 2. 그래프 그리기
fig, ax = plt.subplots(figsize=(8, 6))

# 선과 마커 스타일 지정
ax.plot(sessions, acc_5, marker='o', linestyle='-', linewidth=2, markersize=8, label='5-Shot', color='#95a5a6')
ax.plot(sessions, acc_10, marker='s', linestyle='-', linewidth=2, markersize=8, label='10-Shot', color='#e74c3c') # 최적 10-shot 강조 (빨간색)
ax.plot(sessions, acc_15, marker='^', linestyle='--', linewidth=2, markersize=8, label='15-Shot', color='#3498db')
ax.plot(sessions, acc_20, marker='D', linestyle='--', linewidth=2, markersize=8, label='20-Shot', color='#2980b9')

# 축 및 라벨 설정
ax.set_title('Impact of Shot Size on Incremental Learning Performance', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel(r'Average Accuracy ($A_N$) (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Incremental Sessions', fontsize=12, fontweight='bold')

# 그리드 및 범례
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=11, loc='lower left')

# 값 표시 (선택 사항: 복잡해 보이면 생략 가능하지만 10-shot만 표시)
for i, v in enumerate(acc_10):
    if i == 0:
        ax.text(i, v - 2, f"{v:.2f}", ha='center', fontweight='bold', color='#c0392b')
    else:
        ax.text(i, v + 1.5, f"{v:.2f}", ha='center', fontweight='bold', color='#c0392b')

plt.tight_layout()
plt.savefig('shot_experiment_plot.png', dpi=300)
plt.show()