# 这里主要是绘制折线图用的，用来比较四种状态下（NoMorgan，NoAvalon，NoAll，以及All）
# 某些数值的变化情况
# 这个脚本原本是用来绘制两个特定数据集之间的迁移效果，现在无所谓了应该是
# 后面再说吧
import matplotlib.pyplot as plt

def plot_two_lines(
    group1,
    group2,
    dataset="SIF",
    model="LR",
):
    """
    group1, group2: list of 4 floats
    """

    x_labels = ["NoMorgan", "NoAvalon", "NoAll", "All"]
    x = range(len(x_labels))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(4, 6),   # 不宽，竖向
        sharex=True
    )

    # 第一张图
    axes[0].plot(x, group1, marker="o")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title(f"{dataset}-{model}-(4268->WO2017)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # 第二张图
    axes[1].plot(x, group2, marker="o")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title(f"{dataset}-{model}-(WO2017-4268)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)

    plt.tight_layout()
    plt.show()
    # 保存图像
    fig.savefig(f"A_{dataset}_{model}.png", dpi=300)
# SIF-LR
# group1 = [0.649, 0.588, 0.648, 0.609]
# group2 = [0.594, 0.579, 0.587, 0.547]

# SIF-RF
group1 = [0.647, 0.639, 0.647, 0.642]
group2 = [0.583, 0.588, 0.588, 0.584]

# SIF-XGB
#group1 = [0.626, 0.605, 0.642, 0.622]
#group2 = [0.597, 0.583, 0.579, 0.573]

# SGF-LR
#group1 = [0.700, 0.569, 0.729, 0.587]
#group2 = [0.733, 0.659, 0.716, 0.696]

# SGF-RF
#group1 = [0.736, 0.705, 0.749, 0.711]
#group2 = [0.726, 0.727, 0.700, 0.735]

# SGF-XGB
#group1 = [0.661, 0.650, 0.734, 0.645]
#group2 = [0.673, 0.717, 0.691, 0.712]


plot_two_lines(
    group1,
    group2,
    dataset="SIF",
    model="RF",
)
