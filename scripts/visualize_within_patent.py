#!/usr/bin/env python3
"""
专利内可视化脚本 - Phase 2.1

功能：
1. 单体 vs 二聚体的 SIF/SGF 标签分布对比
2. 单体 vs 二聚体的结构特征对比
3. 统计检验（Mann-Whitney U test）

输入：data/processed/*.csv
输出：outputs/figures/phase2/within_patent/
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualize_within_patent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_processed_csv(csv_path: Path) -> pd.DataFrame:
    """
    加载处理后的 CSV 文件，跳过无效列

    Args:
        csv_path: CSV 文件路径

    Returns:
        DataFrame with valid columns only
    """
    valid_cols = [
        'id', 'SMILES', 'is_monomer', 'is_dimer',
        'is_cyclic', 'has_disulfide_bond',
        'SIF_class', 'SGF_class',
        'SIF_class_min', 'SGF_class_min'
    ]

    try:
        df = pd.read_csv(csv_path, usecols=valid_cols)
        logger.info(f"成功加载 {csv_path.name}: {len(df)} 样本")
        return df
    except Exception as e:
        logger.error(f"加载 CSV 失败 {csv_path}: {e}")
        return None


def plot_monomer_dimer_distributions(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    dpi: int = 300
) -> Dict:
    """
    绘制单体 vs 二聚体的标签分布对比图

    Args:
        df: 数据框
        dataset_name: 数据集名称
        output_dir: 输出目录
        dpi: 图像分辨率

    Returns:
        统计摘要字典
    """
    stats = {}

    # 过滤掉缺失标签 (-1)
    df_sif_valid = df[df['SIF_class_min'] != -1].copy()
    df_sgf_valid = df[df['SGF_class_min'] != -1].copy()

    # === SIF 分布对比 ===
    if len(df_sif_valid) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # 分离单体和二聚体数据
        monomer_sif = df_sif_valid[df_sif_valid['is_monomer'] == True]['SIF_class_min']
        dimer_sif = df_sif_valid[df_sif_valid['is_dimer'] == True]['SIF_class_min']

        # 绘制分组直方图
        bins = np.linspace(
            df_sif_valid['SIF_class_min'].min(),
            df_sif_valid['SIF_class_min'].max(),
            20
        )

        ax.hist(monomer_sif, bins=bins, alpha=0.6, label=f'Monomer (n={len(monomer_sif)})',
                color='steelblue', edgecolor='black')
        ax.hist(dimer_sif, bins=bins, alpha=0.6, label=f'Dimer (n={len(dimer_sif)})',
                color='coral', edgecolor='black')

        # Mann-Whitney U 检验
        if len(monomer_sif) > 0 and len(dimer_sif) > 0:
            stat, p_value = mannwhitneyu(monomer_sif, dimer_sif, alternative='two-sided')
            stats['SIF_mannwhitney_stat'] = float(stat)
            stats['SIF_mannwhitney_pvalue'] = float(p_value)

            # 添加统计信息到图表
            ax.text(0.98, 0.97,
                    f'Mann-Whitney U test\np-value = {p_value:.4f}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)

        ax.set_xlabel('SIF Stability (minutes)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{dataset_name}: Monomer vs Dimer SIF Distribution\n'
                     f'(Valid samples: {len(df_sif_valid)}/{len(df)})',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'{dataset_name}_monomer_dimer_sif_distribution.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存 SIF 分布图: {output_path}")

        # 保存统计数据
        stats['SIF_monomer_mean'] = float(monomer_sif.mean()) if len(monomer_sif) > 0 else None
        stats['SIF_dimer_mean'] = float(dimer_sif.mean()) if len(dimer_sif) > 0 else None
        stats['SIF_monomer_std'] = float(monomer_sif.std()) if len(monomer_sif) > 0 else None
        stats['SIF_dimer_std'] = float(dimer_sif.std()) if len(dimer_sif) > 0 else None

    # === SGF 分布对比 ===
    if len(df_sgf_valid) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # 分离单体和二聚体数据
        monomer_sgf = df_sgf_valid[df_sgf_valid['is_monomer'] == True]['SGF_class_min']
        dimer_sgf = df_sgf_valid[df_sgf_valid['is_dimer'] == True]['SGF_class_min']

        # 绘制分组直方图
        bins = np.linspace(
            df_sgf_valid['SGF_class_min'].min(),
            df_sgf_valid['SGF_class_min'].max(),
            20
        )

        ax.hist(monomer_sgf, bins=bins, alpha=0.6, label=f'Monomer (n={len(monomer_sgf)})',
                color='steelblue', edgecolor='black')
        ax.hist(dimer_sgf, bins=bins, alpha=0.6, label=f'Dimer (n={len(dimer_sgf)})',
                color='coral', edgecolor='black')

        # Mann-Whitney U 检验
        if len(monomer_sgf) > 0 and len(dimer_sgf) > 0:
            stat, p_value = mannwhitneyu(monomer_sgf, dimer_sgf, alternative='two-sided')
            stats['SGF_mannwhitney_stat'] = float(stat)
            stats['SGF_mannwhitney_pvalue'] = float(p_value)

            # 添加统计信息到图表
            ax.text(0.98, 0.97,
                    f'Mann-Whitney U test\np-value = {p_value:.4f}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)

        ax.set_xlabel('SGF Stability (minutes)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{dataset_name}: Monomer vs Dimer SGF Distribution\n'
                     f'(Valid samples: {len(df_sgf_valid)}/{len(df)})',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'{dataset_name}_monomer_dimer_sgf_distribution.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存 SGF 分布图: {output_path}")

        # 保存统计数据
        stats['SGF_monomer_mean'] = float(monomer_sgf.mean()) if len(monomer_sgf) > 0 else None
        stats['SGF_dimer_mean'] = float(dimer_sgf.mean()) if len(dimer_sgf) > 0 else None
        stats['SGF_monomer_std'] = float(monomer_sgf.std()) if len(monomer_sgf) > 0 else None
        stats['SGF_dimer_std'] = float(dimer_sgf.std()) if len(dimer_sgf) > 0 else None

    return stats


def plot_structural_features_comparison(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    dpi: int = 300
):
    """
    绘制单体 vs 二聚体的结构特征对比图

    Args:
        df: 数据框
        dataset_name: 数据集名称
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name}: Structural Features Comparison (Monomer vs Dimer)',
                 fontsize=16, fontweight='bold')

    # 分离单体和二聚体
    monomer_df = df[df['is_monomer'] == True]
    dimer_df = df[df['is_dimer'] == True]

    # 1. 环化率对比（饼图）
    ax1 = axes[0, 0]
    monomer_cyclic_rate = monomer_df['is_cyclic'].mean() * 100 if len(monomer_df) > 0 else 0
    dimer_cyclic_rate = dimer_df['is_cyclic'].mean() * 100 if len(dimer_df) > 0 else 0

    rates = [monomer_cyclic_rate, dimer_cyclic_rate]
    labels = [f'Monomer\n{monomer_cyclic_rate:.1f}%\n(n={len(monomer_df)})',
              f'Dimer\n{dimer_cyclic_rate:.1f}%\n(n={len(dimer_df)})']
    colors = ['steelblue', 'coral']

    ax1.bar(labels, rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Cyclization Rate (%)', fontsize=11)
    ax1.set_title('Cyclization Rate Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # 2. 二硫键含量对比
    ax2 = axes[0, 1]
    monomer_disulfide_rate = monomer_df['has_disulfide_bond'].mean() * 100 if len(monomer_df) > 0 else 0
    dimer_disulfide_rate = dimer_df['has_disulfide_bond'].mean() * 100 if len(dimer_df) > 0 else 0

    rates = [monomer_disulfide_rate, dimer_disulfide_rate]
    labels = [f'Monomer\n{monomer_disulfide_rate:.1f}%\n(n={len(monomer_df)})',
              f'Dimer\n{dimer_disulfide_rate:.1f}%\n(n={len(dimer_df)})']

    ax2.bar(labels, rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Disulfide Bond Rate (%)', fontsize=11)
    ax2.set_title('Disulfide Bond Rate Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)

    # 3. SIF 标签箱线图对比（过滤 -1）
    ax3 = axes[1, 0]
    df_sif_valid = df[df['SIF_class_min'] != -1]
    if len(df_sif_valid) > 0:
        data_to_plot = [
            df_sif_valid[df_sif_valid['is_monomer'] == True]['SIF_class_min'],
            df_sif_valid[df_sif_valid['is_dimer'] == True]['SIF_class_min']
        ]
        bp = ax3.boxplot(data_to_plot, labels=['Monomer', 'Dimer'], patch_artist=True,
                         notch=True, showmeans=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel('SIF Stability (minutes)', fontsize=11)
        ax3.set_title('SIF Stability Comparison', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No valid SIF data',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)

    # 4. SGF 标签箱线图对比（过滤 -1）
    ax4 = axes[1, 1]
    df_sgf_valid = df[df['SGF_class_min'] != -1]
    if len(df_sgf_valid) > 0:
        data_to_plot = [
            df_sgf_valid[df_sgf_valid['is_monomer'] == True]['SGF_class_min'],
            df_sgf_valid[df_sgf_valid['is_dimer'] == True]['SGF_class_min']
        ]
        bp = ax4.boxplot(data_to_plot, labels=['Monomer', 'Dimer'], patch_artist=True,
                         notch=True, showmeans=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_ylabel('SGF Stability (minutes)', fontsize=11)
        ax4.set_title('SGF Stability Comparison', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No valid SGF data',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)

    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_structural_features_comparison.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存结构特征对比图: {output_path}")


def process_dataset(csv_path: Path, output_base_dir: Path, dpi: int = 300):
    """
    处理单个数据集的可视化

    Args:
        csv_path: CSV 文件路径
        output_base_dir: 输出基础目录
        dpi: 图像分辨率
    """
    dataset_name = csv_path.stem.replace('_processed', '')
    logger.info(f"开始处理数据集: {dataset_name}")

    # 加载数据
    df = load_processed_csv(csv_path)
    if df is None or len(df) == 0:
        logger.warning(f"跳过数据集 {dataset_name}：数据加载失败或为空")
        return

    # 创建数据集专属输出目录
    output_dir = output_base_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 1. 绘制单体 vs 二聚体分布对比
    stats = plot_monomer_dimer_distributions(df, dataset_name, output_dir, dpi)

    # 2. 绘制结构特征对比
    plot_structural_features_comparison(df, dataset_name, output_dir, dpi)

    # 3. 保存统计摘要
    stats['dataset_name'] = dataset_name
    stats['total_samples'] = int(len(df))
    stats['monomer_count'] = int(df['is_monomer'].sum())
    stats['dimer_count'] = int(df['is_dimer'].sum())
    stats['cyclic_count'] = int(df['is_cyclic'].sum())
    stats['disulfide_count'] = int(df['has_disulfide_bond'].sum())
    stats['SIF_valid_count'] = int((df['SIF_class_min'] != -1).sum())
    stats['SGF_valid_count'] = int((df['SGF_class_min'] != -1).sum())

    stats_path = output_dir / f'{dataset_name}_statistical_summary.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"已保存统计摘要: {stats_path}")

    logger.info(f"数据集 {dataset_name} 处理完成\n")


def main():
    parser = argparse.ArgumentParser(
        description='专利内可视化脚本 - Phase 2.1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('data/processed'),
        help='输入目录（包含处理后的 CSV 文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/figures/phase2/within_patent'),
        help='输出目录'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='图像分辨率'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("开始专利内可视化分析 - Phase 2.1")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"图像分辨率: {args.dpi} DPI")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有处理后的 CSV 文件
    csv_files = sorted(args.input_dir.glob('*_processed.csv'))

    if not csv_files:
        logger.error(f"在 {args.input_dir} 中未找到任何 *_processed.csv 文件")
        return

    logger.info(f"找到 {len(csv_files)} 个数据集待处理")

    # 处理每个数据集
    for csv_path in csv_files:
        process_dataset(csv_path, args.output_dir, args.dpi)

    logger.info("=" * 60)
    logger.info("专利内可视化分析完成！")
    logger.info(f"所有结果已保存到: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
