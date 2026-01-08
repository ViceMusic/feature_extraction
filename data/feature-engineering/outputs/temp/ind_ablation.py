import json
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 用户需要改的部分（复制实验时只改这里）
# =========================================================

# ====== A. 选择要对比的 JSON 文件（顺序 = 图例顺序）======
# 示例：
# 1) 单特征 baseline
# 2) Morgan 锚点加法
# 3) Remove-one 消融
SELECTED_JSONS = [
    "Final_Results_Morgan(1024).json",
    "Final_Results_Avalon(512).json",
    "Final_Results_ChemBERTa(384).json",
]

# Morgan锚点绘制
SELECTED_JSONS = [
    "Final_Results_Morgan(1024).json",
    "Final_Results_Morgan(1024)_Avalon(512).json",
    "Final_Results_Morgan(1024)_ChemBERTa(384).json",
    "Final_Results_Morgan(1024)_Avalon(512)_ChemBERTa(384).json",
]

SELECTED_JSONS = [
    "Final_Results_Morgan(1024)_Avalon(512)_ChemBERTa(384).json",
    "Final_Results_Avalon(512)_ChemBERTa(384).json",
    "Final_Results_Morgan(1024)_ChemBERTa(384).json",
    "Final_Results_Morgan(1024)_Avalon(512).json",
]



# ====== B. 选择要画的 case ======
CASES = [
    "SIF_lr", "SIF_rf", "SIF_xgb",
    "SGF_lr", "SGF_rf", "SGF_xgb"
]

# ====== C. 图的总标题（会显示在最上方）======
FIG_TITLE = "Single-Feature Baseline Comparison"

FIG_TITLE = "Morgan-Centered Feature Addition"

FIG_TITLE = "Remove-One Ablation Study"

# ====== JSON 文件所在目录 ======
JSON_DIR = "./verfiy_results"

# =========================================================
# 固定配置（一般不用动）
# =========================================================

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
METRIC_LABELS = [m.upper() for m in METRICS]

# 使用明显区分的调色板
COLOR_MAP = plt.get_cmap("tab10")

# =========================================================
# 1. 读取指定的 JSON
# =========================================================

results = {}  # {json_name: data}

for fname in SELECTED_JSONS:
    path = os.path.join(JSON_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, "r") as f:
        results[fname] = json.load(f)["ind"]

labels = [os.path.splitext(f)[0] for f in SELECTED_JSONS]
num_models = len(labels)

# =========================================================
# 2. 雷达图角度
# =========================================================

angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
angles += angles[:1]  # 闭合

# =========================================================
# 3. 创建画布
# =========================================================

n_cases = len(CASES)
n_cols = 3
n_rows = int(np.ceil(n_cases / n_cols))

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(6 * n_cols, 5 * n_rows),
    subplot_kw=dict(polar=True)
)

axes = axes.flatten()

# =========================================================
# 4. 绘制每个 case
# =========================================================

for i, case in enumerate(CASES):
    ax = axes[i]
    ax.set_title(case, fontsize=13, pad=12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.4)

    for idx, label in enumerate(labels):
        color = COLOR_MAP(idx % 10)
        values = [results[label + ".json" if label + ".json" in results else label][case][m] for m in METRICS]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, color=color, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

# 删除多余子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# =========================================================
# 5. 图例 & 标题
# =========================================================

# =========================
# 手动构造 legend handles（关键修复）
# =========================
legend_handles = []

from matplotlib.lines import Line2D

for idx, label in enumerate(labels):
    color = COLOR_MAP(idx % 10)
    handle = Line2D(
        [0], [0],
        color=color,
        linewidth=2,
        label=label
    )
    legend_handles.append(handle)

fig.legend(
    handles=legend_handles,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=min(4, num_models),
    frameon=False,
    fontsize=11
)


fig.suptitle(FIG_TITLE, fontsize=16, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.88])


# =========================================================
# 6. 保存
# =========================================================

os.makedirs("figures", exist_ok=True)
save_path = f"figures/{FIG_TITLE.replace(' ', '_')}.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved to {save_path}")

plt.show()
