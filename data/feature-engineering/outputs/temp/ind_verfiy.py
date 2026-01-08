import json
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 基本配置
# =========================
json_dir = "./verfiy_results"   # 存放多个 json 的目录
save_dir = "./figures"
os.makedirs(save_dir, exist_ok=True)

cases = [
    "SIF_lr", "SIF_rf", "SIF_xgb",
    "SGF_lr", "SGF_rf", "SGF_xgb"
]

metrics = ["accuracy", "precision", "recall", "f1", "auc"]

# =========================
# 2. 读取所有 JSON
# =========================
all_results = {}  # {task_name: data["ind"]}

for fname in sorted(os.listdir(json_dir)):
    if fname.endswith(".json"):
        task_name = os.path.splitext(fname)[0]
        with open(os.path.join(json_dir, fname), "r") as f:
            all_results[task_name] = json.load(f)["ind"]

task_names = list(all_results.keys())
num_tasks = len(task_names)

print(f"[INFO] Loaded {num_tasks} tasks:")
for t in task_names:
    print("  -", t)

# =========================
# 3. 雷达图角度
# =========================
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合

# =========================
# 4. 全局颜色映射（关键）
# =========================
cmap = plt.cm.get_cmap("tab10", num_tasks)
task_color = {
    task: cmap(i)
    for i, task in enumerate(task_names)
}

# =========================
# 5. 创建画布（2 x 3）
# =========================
fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(18, 10),
    subplot_kw=dict(polar=True)
)
axes = axes.flatten()

# =========================
# 6. 绘制每一个 case
# =========================
for i, case in enumerate(cases):
    ax = axes[i]
    ax.set_title(case, fontsize=13, pad=12)

    # 坐标轴设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.4)

    # 绘制每个 task
    for task in task_names:
        values = [all_results[task][case][m] for m in metrics]
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            color=task_color[task],
            label=task
        )
        ax.fill(
            angles,
            values,
            color=task_color[task],
            alpha=0.10
        )

# =========================
# 7. 全局 Legend（从真实对象生成）
# =========================
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=min(4, num_tasks),
    frameon=False,
    fontsize=11
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# =========================
# 8. 保存 & 显示
# =========================
save_path = os.path.join(save_dir, "radar_comparison_all_cases.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"[INFO] Figure saved to: {save_path}")

plt.show()
