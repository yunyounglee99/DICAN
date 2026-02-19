import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 공통 스타일 설정
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']

# ==========================================
# 2. 데이터 준비
# ==========================================
grades = ['Grade 0', 'Grade 1', 'Grade 2\n(Pred)', 'Grade 3\n(GT)', 'Grade 4']
grades_after = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3\n(Pred & GT)', 'Grade 4']

prob_before = [0.072, 0.093, 0.334, 0.324, 0.176]
prob_after = [0.000, 0.000, 0.000, 1.000, 0.000]

color_before = ['#bdc3c7', '#bdc3c7', '#bdc3c7', '#bdc3c7', '#bdc3c7'] 
color_after = ['#bdc3c7', '#bdc3c7', '#bdc3c7', "#33ff33", '#bdc3c7']

# ==========================================
# 3. 그래프 생성 함수 (막대 간격 0, 딱 붙게 설정)
# ==========================================
def create_prob_chart(x_labels, probabilities, colors, filename):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    ax.set_facecolor('white')

    x = np.arange(len(x_labels))
    
    # [핵심 변경] width=1.0으로 설정하여 막대 간의 간격을 없앰
    # 막대가 붙어있으므로 구분을 위해 테두리(edgecolor)를 하얀색으로 굵게 설정
    bars = ax.bar(x, probabilities, color=colors, width=1.0, edgecolor='black', linewidth=0.5)

    # 모든 테두리(축 선) 숨기기
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Y축 눈금 및 라벨 숨기기
    ax.yaxis.set_visible(False)
    
    # X축 눈금선(tick)은 숨기고 라벨만 남기기
    ax.tick_params(axis='x', length=0, labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight='bold')

    # 막대 위에 정확한 확률 값 표시
    for bar in bars:
        height = bar.get_height()
        alpha_val = 1.0 if height > 0 else 0.4
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points 위로
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', alpha=alpha_val)

    # Y축 상단 여유 공간 (텍스트 잘림 방지)
    ax.set_ylim(0, max(probabilities) * 1.25)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True) # 배경 투명
    plt.close()

# ==========================================
# 4. 각각의 그래프 생성 및 저장
# ==========================================
create_prob_chart(grades, prob_before, color_before, 'prob_before_attached.png')
create_prob_chart(grades_after, prob_after, color_after, 'prob_after_attached.png')

print("간격이 없는 딱 붙은 그래프가 성공적으로 생성되었습니다!")