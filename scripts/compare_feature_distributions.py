#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征分布对比可视化脚本

加载两个 NPZ 特征文件，生成对比分析可视化图表。
在同一子图中绘制两个数据集的特征分布，便于直观比较。

用法：
    python scripts/compare_feature_distributions.py \\
        --dataset1 outputs/features/dataset1.npz \\
        --dataset2 outputs/features/dataset2.npz \\
        --output_dir outputs/figures/ \\
        --dpi 300 \\
        --format png
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    配置日志系统。

    Args:
        log_level (int): 日志级别。默认 logging.INFO。
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("comparison_visualization.log"),
        ],
    )


def load_npz_file(npz_path: Path) -> tuple:
    """
    加载 NPZ 文件。

    Args:
        npz_path (Path): NPZ 文件路径。

    Returns:
        tuple: (X, y_sif, y_sgf, feature_names) 元组。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 必需的键不存在。
    """
    logger = logging.getLogger(__name__)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ 文件不存在: {npz_path}")

    try:
        npz = np.load(npz_path, allow_pickle=True)

        required_keys = ['X', 'y_sif', 'y_sgf', 'feature_names']
        missing_keys = [k for k in required_keys if k not in npz]

        if missing_keys:
            raise ValueError(f"NPZ 文件缺失必需的键: {', '.join(missing_keys)}")

        X = npz['X']
        y_sif = npz['y_sif']
        y_sgf = npz['y_sgf']
        feature_names = npz['feature_names']

        logger.info(
            f"成功加载 NPZ 文件: {npz_path.name} "
            f"({X.shape[0]} 个样本, {X.shape[1]} 个特征)"
        )

        return X, y_sif, y_sgf, feature_names

    except Exception as e:
        logger.error(f"加载 NPZ 文件出错: {e}")
        raise


def get_feature_by_name(
    feature_name: str,
    X: np.ndarray,
    feature_names: np.ndarray
) -> Optional[np.ndarray]:
    """
    按名称从特征矩阵中获取特征向量。

    Args:
        feature_name (str): 特征名称。
        X (np.ndarray): 特征矩阵，shape: (n_samples, n_features)。
        feature_names (np.ndarray): 特征名称列表。

    Returns:
        np.ndarray | None: 特征向量，若不存在则返回 None。
    """
    if feature_name in feature_names:
        idx = list(feature_names).index(feature_name)
        return X[:, idx]
    return None


def plot_comparison_feature_distributions(
    X1: np.ndarray,
    feature_names1: np.ndarray,
    dataset_name1: str,
    X2: np.ndarray,
    feature_names2: np.ndarray,
    dataset_name2: str,
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制两个数据集的特征分布对比图。

    在同一个 2×2 子图布局中，同时绘制两个数据集的特征分布。
    每个数据集使用不同的颜色，包括直方图、KDE 曲线、均值和中位数线。

    Args:
        X1 (np.ndarray): 数据集1的特征矩阵，shape: (n_samples, n_features)。
        feature_names1 (np.ndarray): 数据集1的特征名称列表。
        dataset_name1 (str): 数据集1的名称。
        X2 (np.ndarray): 数据集2的特征矩阵，shape: (n_samples, n_features)。
        feature_names2 (np.ndarray): 数据集2的特征名称列表。
        dataset_name2 (str): 数据集2的名称。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。默认 300。
        format (str): 输出格式。默认 "png"。

    Returns:
        Path: 保存的图表路径。

    Raises:
        ValueError: 必需的特征不存在。
    """
    logger = logging.getLogger(__name__)

    features_to_plot = [
        ('PC_MolWt', 'Molecular Weight (g/mol)'),
        ('PC_LogP', 'LogP (Lipophilicity)'),
        ('PC_HBA', 'Hydrogen Bond Acceptors'),
        ('PC_HBD', 'Hydrogen Bond Donors'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Feature Distribution Comparison\n{dataset_name1} vs {dataset_name2}",
        fontsize=16, fontweight='bold'
    )

    axes = axes.flatten()

    # 颜色设置：数据集1使用蓝色系，数据集2使用红色系
    colors_dataset1 = {
        'histogram': 'skyblue',
        'kde': 'blue',
        'mean': 'green',
        'median': 'orange'
    }
    colors_dataset2 = {
        'histogram': 'lightcoral',
        'kde': 'red',
        'mean': 'darkgreen',
        'median': 'darkorange'
    }

    for ax, (feature_name, label) in zip(axes, features_to_plot):
        # 获取数据集1的特征数据
        feature_data1 = get_feature_by_name(feature_name, X1, feature_names1)
        # 获取数据集2的特征数据
        feature_data2 = get_feature_by_name(feature_name, X2, feature_names2)

        if feature_data1 is None or feature_data2 is None:
            if feature_data1 is None:
                logger.warning(f"特征 {feature_name} 在数据集1中不存在")
            if feature_data2 is None:
                logger.warning(f"特征 {feature_name} 在数据集2中不存在")
            ax.text(
                0.5, 0.5, f"Feature {feature_name} not found",
                ha='center', va='center', transform=ax.transAxes
            )
            continue

        # 确定共同的数据范围
        x_min = min(feature_data1.min(), feature_data2.min())
        x_max = max(feature_data1.max(), feature_data2.max())
        x_range = np.linspace(x_min, x_max, 200)

        # 绘制数据集1的直方图
        ax.hist(
            feature_data1, bins=30, alpha=0.5, color=colors_dataset1['histogram'],
            edgecolor='blue', density=True, label=f'{dataset_name1} (Histogram)'
        )

        # 绘制数据集1的 KDE 曲线
        try:
            kde1 = gaussian_kde(feature_data1)
            ax.plot(
                x_range, kde1(x_range), color=colors_dataset1['kde'],
                linewidth=2.5, label=f'{dataset_name1} (KDE)'
            )
        except Exception as e:
            logger.warning(f"计算数据集1的KDE失败: {e}")

        # 绘制数据集1的均值和中位数线
        mean_val1 = np.mean(feature_data1)
        median_val1 = np.median(feature_data1)
        ax.axvline(
            mean_val1, color=colors_dataset1['mean'], linestyle='--',
            linewidth=2, label=f'{dataset_name1} Mean: {mean_val1:.2f}'
        )
        ax.axvline(
            median_val1, color=colors_dataset1['median'], linestyle=':',
            linewidth=2, label=f'{dataset_name1} Median: {median_val1:.2f}'
        )

        # 绘制数据集2的直方图
        ax.hist(
            feature_data2, bins=30, alpha=0.5, color=colors_dataset2['histogram'],
            edgecolor='red', density=True, label=f'{dataset_name2} (Histogram)'
        )

        # 绘制数据集2的 KDE 曲线
        try:
            kde2 = gaussian_kde(feature_data2)
            ax.plot(
                x_range, kde2(x_range), color=colors_dataset2['kde'],
                linewidth=2.5, label=f'{dataset_name2} (KDE)'
            )
        except Exception as e:
            logger.warning(f"计算数据集2的KDE失败: {e}")

        # 绘制数据集2的均值和中位数线
        mean_val2 = np.mean(feature_data2)
        median_val2 = np.median(feature_data2)
        ax.axvline(
            mean_val2, color=colors_dataset2['mean'], linestyle='--',
            linewidth=2, label=f'{dataset_name2} Mean: {mean_val2:.2f}'
        )
        ax.axvline(
            median_val2, color=colors_dataset2['median'], linestyle=':',
            linewidth=2, label=f'{dataset_name2} Median: {median_val2:.2f}'
        )

        # 计算统计信息
        std_val1 = np.std(feature_data1)
        skew_val1 = (
            3 * (mean_val1 - median_val1) / std_val1 if std_val1 > 0 else 0
        )

        std_val2 = np.std(feature_data2)
        skew_val2 = (
            3 * (mean_val2 - median_val2) / std_val2 if std_val2 > 0 else 0
        )

        # 添加统计信息框（左上角：数据集1，右上角：数据集2）
        stats_text1 = (
            f"{dataset_name1}\n"
            f"N={len(feature_data1)}\n"
            f"μ={mean_val1:.2f}\n"
            f"σ={std_val1:.2f}\n"
            f"Skew={skew_val1:.2f}"
        )
        ax.text(
            0.02, 0.97, stats_text1, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        )

        stats_text2 = (
            f"{dataset_name2}\n"
            f"N={len(feature_data2)}\n"
            f"μ={mean_val2:.2f}\n"
            f"σ={std_val2:.2f}\n"
            f"Skew={skew_val2:.2f}"
        )
        ax.text(
            0.98, 0.97, stats_text2, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7)
        )

        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper center', fontsize=7.5, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = (
        output_dir /
        f"{Path(dataset_name1).stem}_vs_{Path(dataset_name2).stem}_"
        f"feature_distributions_comparison.{format}"
    )
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存特征分布对比图: {output_path}")
    plt.close()

    return output_path


def main() -> int:
    """
    主函数：生成两个数据集的特征分布对比可视化。

    Returns:
        int: 程序退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(
        description="生成两个数据集的特征分布对比可视化图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/compare_feature_distributions.py "
            "--dataset1 outputs/features/dataset1.npz "
            "--dataset2 outputs/features/dataset2.npz "
            "--output_dir outputs/figures/\n"
        ),
    )

    parser.add_argument(
        "--dataset1",
        type=Path,
        required=True,
        help="第一个数据集的 NPZ 文件路径（必需）",
    )
    parser.add_argument(
        "--dataset2",
        type=Path,
        required=True,
        help="第二个数据集的 NPZ 文件路径（必需）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/figures"),
        help="输出可视化图表的目录（默认: outputs/figures）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图像分辨率，DPI（默认: 300）",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="输出图像格式（默认: png）",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别（默认: INFO）",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("开始 SIF/SGF 特征分布对比可视化")
    logger.info("=" * 70)
    logger.info(f"数据集1: {args.dataset1.resolve()}")
    logger.info(f"数据集2: {args.dataset2.resolve()}")
    logger.info(f"输出目录: {args.output_dir.resolve()}")
    logger.info(f"图像分辨率: {args.dpi} DPI")
    logger.info(f"输出格式: {args.format}")

    # 验证输入文件
    if not args.dataset1.exists():
        logger.error(f"数据集1文件不存在: {args.dataset1}")
        return 2

    if not args.dataset2.exists():
        logger.error(f"数据集2文件不存在: {args.dataset2}")
        return 2

    # 创建输出目录
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已创建: {args.output_dir.resolve()}")
    except OSError as e:
        logger.error(f"创建输出目录失败: {e}")
        return 2

    # 加载两个数据集
    try:
        X1, y_sif1, y_sgf1, feature_names1 = load_npz_file(args.dataset1)
        X2, y_sif2, y_sgf2, feature_names2 = load_npz_file(args.dataset2)
    except FileNotFoundError as e:
        logger.error(f"文件加载失败: {e}")
        return 2
    except ValueError as e:
        logger.error(f"NPZ 文件格式错误: {e}")
        return 2

    # 生成对比可视化
    try:
        logger.info("开始生成特征分布对比图...")
        plot_comparison_feature_distributions(
            X1=X1,
            feature_names1=feature_names1,
            dataset_name1=args.dataset1.stem,
            X2=X2,
            feature_names2=feature_names2,
            dataset_name2=args.dataset2.stem,
            output_dir=args.output_dir,
            dpi=args.dpi,
            format=args.format
        )

        logger.info("=" * 70)
        logger.info("特征分布对比可视化生成完成")
        logger.info("=" * 70)
        logger.info(f"输出目录: {args.output_dir.resolve()}")
        logger.info("对比分析完成！")
        return 0

    except Exception as e:
        logger.error(f"生成可视化时出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

