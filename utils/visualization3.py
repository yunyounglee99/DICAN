import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_dican_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # 설정: 박스 스타일
    box_style = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', linewidth=1.5)
    arrow_style = dict(arrowstyle="->", color="black", linewidth=2, mutation_scale=20)

    # 1. Input Image
    ax.add_patch(patches.Rectangle((2, 15), 10, 20, facecolor='#D5E8D4', edgecolor='#82B366', linewidth=2))
    ax.text(7, 25, "Input Image\n(224x224x3)", ha='center', va='center', fontweight='bold')

    # 2. Backbone (ResNet50)
    ax.add_patch(patches.Rectangle((18, 15), 12, 20, facecolor='#DAE8FC', edgecolor='#6C8EBF', linewidth=2))
    ax.text(24, 25, "Backbone\n(ResNet50)", ha='center', va='center', fontweight='bold')
    ax.text(24, 12, "Fixed in Inc. Mode", ha='center', color='red', fontsize=9)

    # 3. Projector (1x1 Conv)
    ax.add_patch(patches.Rectangle((36, 17), 10, 16, facecolor='#FFF2CC', edgecolor='#D6B656', linewidth=2))
    ax.text(41, 25, "Projector\n(Alignment)", ha='center', va='center', fontweight='bold')
    ax.text(41, 14, "Trainable\n(Domain-wise)", ha='center', color='blue', fontsize=9)

    # 4. Prototype Bank (Spatial Similarity)
    # 4개 병변(EX, HE, MA, SE) 표현
    ax.add_patch(patches.Rectangle((52, 10), 15, 30, facecolor='#F8CECC', edgecolor='#B85450', linewidth=2, linestyle='--'))
    ax.text(59.5, 37, "Prototype Bank", ha='center', va='center', fontweight='bold')
    
    protos = ["EX", "HE", "MA", "SE"]
    for i, p in enumerate(protos):
        ax.add_patch(patches.Circle((59.5, 30 - i*6), 2, facecolor='#FAD7A0', edgecolor='#E67E22'))
        ax.text(59.5, 30 - i*6, p, ha='center', va='center', fontsize=8, fontweight='bold')

    # 5. LSE Pooling (Aggregation)
    ax.add_patch(patches.Rectangle((73, 20), 8, 10, facecolor='#E1D5E7', edgecolor='#9673A6', linewidth=2))
    ax.text(77, 25, "LSE\nPooling", ha='center', va='center', fontweight='bold', fontsize=9)

    # 6. Reasoning Head
    ax.add_patch(patches.Rectangle((87, 15), 10, 20, facecolor='#F5F5F5', edgecolor='#333333', linewidth=2))
    ax.text(92, 25, "Reasoning\nHead\n(MLP)", ha='center', va='center', fontweight='bold')
    ax.text(92, 12, "Frozen in Inc.", ha='center', color='gray', fontsize=9)

    # --- 화살표 연결 ---
    # Input -> Backbone
    ax.annotate("", xy=(18, 25), xytext=(12, 25), arrowprops=arrow_style)
    # Backbone -> Projector
    ax.annotate("", xy=(36, 25), xytext=(30, 25), arrowprops=arrow_style)
    ax.text(33, 27, "7x7x2048", ha='center', fontsize=8)
    # Projector -> Prototype
    ax.annotate("", xy=(52, 25), xytext=(46, 25), arrowprops=arrow_style)
    # Prototype -> LSE
    ax.annotate("", xy=(73, 25), xytext=(67, 25), arrowprops=arrow_style)
    ax.text(70, 27, "7x7x4\n(Maps)", ha='center', fontsize=8)
    # LSE -> Head
    ax.annotate("", xy=(87, 25), xytext=(81, 25), arrowprops=arrow_style)
    ax.text(84, 27, "1x4\n(Scores)", ha='center', fontsize=8)

    # 하단 캡션
    plt.title("DICAN: Domain-Incremental Concept Bottleneck Alignment Network", fontsize=18, fontweight='bold', pad=30)
    plt.text(50, 2, "*EX: Exudates, HE: Hemorrhages, MA: Microaneurysms, SE: Soft Exudates", ha='center', style='italic', fontsize=10)

    plt.tight_layout()
    plt.savefig('dican_model_diagram.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_dican_architecture()