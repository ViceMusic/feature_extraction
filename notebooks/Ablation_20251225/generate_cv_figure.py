# 该脚本的主要工作内容为，配合计算以后的均值json，绘制热力图
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_matrix(data, data_type, model, metric_type):
    """
    返回：
        datasets: list[str]
        matrix: np.ndarray (n_dataset, 5)
    """
    datasets = []
    rows = []

    model_block = data[data_type][model]

    for dataset, values in model_block.items():
        metric_list = values.get(metric_type, [])
        if not metric_list:
            continue

        datasets.append(dataset)
        row = [metric_list[0][m] for m in METRICS]
        rows.append(row)

    return datasets, np.array(rows)
def plot_cv_heatmaps(
    json_paths,
    labels,
    ref_index,
    data_type,
    model,
    metric_type,
    save_dir="cv_figure",
):
    os.makedirs(save_dir, exist_ok=True)

    datas = [load_json(p) for p in json_paths]

    ref_datasets, ref_mat = extract_matrix(
        datas[ref_index], data_type, model, metric_type
    )

    n_plots = len(json_paths) - 1

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(3.0 * n_plots, 3.0),   # 稍微放大图本体
        constrained_layout=True,
    )

    if n_plots == 1:
        axes = [axes]

    plot_id = 0
    for i, (data, label) in enumerate(zip(datas, labels)):
        if i == ref_index:
            continue

        datasets, mat = extract_matrix(data, data_type, model, metric_type)

        idx = [datasets.index(d) for d in ref_datasets]
        mat = mat[idx]

        diff = mat - ref_mat if metric_type == "means" else ref_mat - mat

        ax = axes[plot_id]

        # 🔹 新增处理：将 NaN / Inf 显示为白色
        mask = ~np.isfinite(diff)  # True 表示遮罩
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        cmap.set_bad("white")

        hm = sns.heatmap(
            diff,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-0.1,
            vmax=0.1,
            square=True,
            annot=True,
            fmt=".4f",
            annot_kws={"size": 4},
            xticklabels=METRICS,
            yticklabels=ref_datasets,
            mask=mask,               # 🔹 遮罩 NaN / Inf
            cbar=True,
            cbar_kws={
                "shrink": 0.55,
                "aspect": 25,
            },
        )

        ax.set_title(
            f"{label} − {labels[ref_index]}",
            fontsize=5,
            pad=1,
        )

        # 明显缩小坐标轴字体
        ax.tick_params(axis="x", labelsize=5, rotation=45, pad=1)
        ax.tick_params(axis="y", labelsize=5, pad=1)

        # colorbar 字体同步缩小
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        plot_id += 1

    # 🔹 调整 suptitle 位置，使其靠近子图
    fig.suptitle(
        f"===={data_type} | {model.upper()} | {metric_type}-{'(no-with)' if metric_type=='means' else '(with-no)'}====",
        fontsize=8,
        y=0.85,  # 原本 1.02，改成 0.98 更贴近子图
    )

    save_path = Path(save_dir) / f"{data_type}_{model}_{metric_type}_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")


json_files = [
    "NoMorgan_averaged.json",
    "NoAvalon_averaged.json",
    "NoAll_averaged.json",
    "all_averaged.json",
]

labels = ["NoMorgan", "NoAvalon", "NoAll", "Complete"]

# 可选参数集合
data_types = ["SIF", "SGF"]
models = ["lr", "rf", "xgb"]
metric_types = ["means", "stds"]

# 固定参数
ref_index = 3          # D 作为参考
json_paths = json_files
labels = labels

for data_type in data_types:
    for model in models:
        for metric_type in metric_types:
            print(
                f"Plotting: data_type={data_type}, "
                f"model={model}, metric_type={metric_type}"
            )

            plot_cv_heatmaps(
                json_paths=json_paths,
                labels=labels,
                ref_index=ref_index,
                data_type=data_type,
                model=model,
                metric_type=metric_type,
            )

