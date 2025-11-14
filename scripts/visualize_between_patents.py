#!/usr/bin/env python3
"""
专利间可视化脚本 - Phase 2.2

功能：
1. 降维可视化（PCA/t-SNE）- 用颜色区分专利来源，点大小表示半衰期
2. 标签分布对比（小提琴图、箱线图）
3. 特征统计对比（样本量、单体率、环化率、二硫键率）
4. 统计检验（ANOVA/Kruskal-Wallis）

输入：outputs/features/*.npz + data/processed/*.csv
输出：outputs/figures/phase2/between_patents/
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal, f_oneway

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualize_between_patents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_all_features(features_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    加载所有数据集的特征并合并

    Args:
        features_dir: 特征文件目录

    Returns:
        (X_all, y_sif_all, y_sgf_all, source_labels)
    """
    npz_files = sorted(features_dir.glob('*_processed.npz'))

    if not npz_files:
        logger.error(f"在 {features_dir} 中未找到任何 *_processed.npz 文件")
        return None, None, None, None

    X_list = []
    y_sif_list = []
    y_sgf_list = []
    source_list = []

    for npz_path in npz_files:
        dataset_name = npz_path.stem.replace('_processed', '')
        logger.info(f"加载特征文件: {npz_path.name}")

        npz = np.load(npz_path, allow_pickle=True)
        X = npz['X']
        y_sif = npz['y_sif']
        y_sgf = npz['y_sgf']

        X_list.append(X)
        y_sif_list.append(y_sif)
        y_sgf_list.append(y_sgf)
        source_list.extend([dataset_name] * len(X))

        logger.info(f"  样本数: {len(X)}, 特征数: {X.shape[1]}")

    # 合并所有数据
    X_all = np.vstack(X_list)
    y_sif_all = np.hstack(y_sif_list)
    y_sgf_all = np.hstack(y_sgf_list)

    logger.info(f"合并后总样本数: {len(X_all)}")

    return X_all, y_sif_all, y_sgf_all, source_list


def load_all_csv_data(processed_dir: Path) -> pd.DataFrame:
    """
    加载所有处理后的 CSV 数据并合并

    Args:
        processed_dir: 处理后数据目录

    Returns:
        合并后的 DataFrame
    """
    valid_cols = [
        'id', 'SMILES', 'is_monomer', 'is_dimer',
        'is_cyclic', 'has_disulfide_bond',
        'SIF_class', 'SGF_class',
        'SIF_class_min', 'SGF_class_min'
    ]

    csv_files = sorted(processed_dir.glob('*_processed.csv'))
    df_list = []

    for csv_path in csv_files:
        dataset_name = csv_path.stem.replace('_processed', '')
        df = pd.read_csv(csv_path, usecols=valid_cols)
        df['dataset'] = dataset_name
        df_list.append(df)
        logger.info(f"加载 CSV: {csv_path.name} ({len(df)} 样本)")

    df_all = pd.concat(df_list, ignore_index=True)
    logger.info(f"合并后总样本数: {len(df_all)}")

    return df_all


def plot_dimensionality_reduction(
    X: np.ndarray,
    y_sif: np.ndarray,
    source_labels: List[str],
    output_dir: Path,
    dpi: int = 300
):
    """
    绘制降维可视化（PCA 和 t-SNE）

    Args:
        X: 特征矩阵
        y_sif: SIF 标签（用于点大小）
        source_labels: 数据集来源标签（用于颜色）
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 过滤无效的 SIF 标签（-1）用于点大小映射
    sif_sizes = np.where(y_sif == -1, 30, y_sif / 10 + 30)  # 默认大小 30，有效标签按比例缩放

    # 获取唯一数据集列表并分配颜色
    unique_sources = sorted(set(source_labels))
    color_palette = sns.color_palette("husl", len(unique_sources))
    source_to_color = {src: color_palette[i] for i, src in enumerate(unique_sources)}
    colors = [source_to_color[src] for src in source_labels]

    # === PCA 降维到 2D ===
    logger.info("执行 PCA 降维...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 为每个数据集单独绘制散点（以便图例清晰）
    for src in unique_sources:
        mask = [s == src for s in source_labels]
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[source_to_color[src]],
            s=sif_sizes[mask],
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5,
            label=src
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA 2D Projection by Patent Source\n(Point size = SIF stability)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'pca_2d_by_patent.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存 PCA 图: {output_path}")

    # === t-SNE 降维到 2D ===
    logger.info("执行 t-SNE 降维（这可能需要几分钟）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))

    for src in unique_sources:
        mask = [s == src for s in source_labels]
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=[source_to_color[src]],
            s=sif_sizes[mask],
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5,
            label=src
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE 2D Projection by Patent Source\n(Point size = SIF stability)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'tsne_2d_by_patent.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存 t-SNE 图: {output_path}")


def plot_label_distribution_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300
):
    """
    绘制跨数据集的标签分布对比（小提琴图 + 箱线图）

    Args:
        df: 合并的数据框
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    # 过滤有效标签
    df_sif = df[df['SIF_class_min'] != -1].copy()
    df_sgf = df[df['SGF_class_min'] != -1].copy()

    # === SIF 小提琴图 ===
    if len(df_sif) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))

        sns.violinplot(
            data=df_sif,
            x='dataset',
            y='SIF_class_min',
            ax=ax,
            palette='husl',
            inner='box'
        )

        ax.set_xlabel('Patent Dataset', fontsize=12)
        ax.set_ylabel('SIF Stability (minutes)', fontsize=12)
        ax.set_title('SIF Stability Distribution Across Patents (Violin Plot)',
                     fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Kruskal-Wallis 检验
        groups = [group['SIF_class_min'].values for name, group in df_sif.groupby('dataset')]
        if len(groups) > 1:
            stat, p_value = kruskal(*groups)
            ax.text(0.98, 0.97,
                    f'Kruskal-Wallis test\np-value = {p_value:.4e}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)

        plt.tight_layout()
        output_path = output_dir / 'violin_plot_sif.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存 SIF 小提琴图: {output_path}")

    # === SGF 小提琴图 ===
    if len(df_sgf) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))

        sns.violinplot(
            data=df_sgf,
            x='dataset',
            y='SGF_class_min',
            ax=ax,
            palette='husl',
            inner='box'
        )

        ax.set_xlabel('Patent Dataset', fontsize=12)
        ax.set_ylabel('SGF Stability (minutes)', fontsize=12)
        ax.set_title('SGF Stability Distribution Across Patents (Violin Plot)',
                     fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Kruskal-Wallis 检验
        groups = [group['SGF_class_min'].values for name, group in df_sgf.groupby('dataset')]
        if len(groups) > 1:
            stat, p_value = kruskal(*groups)
            ax.text(0.98, 0.97,
                    f'Kruskal-Wallis test\np-value = {p_value:.4e}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)

        plt.tight_layout()
        output_path = output_dir / 'violin_plot_sgf.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存 SGF 小提琴图: {output_path}")

    # === 箱线图对比 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # SIF 箱线图
    if len(df_sif) > 0:
        sns.boxplot(
            data=df_sif,
            x='dataset',
            y='SIF_class_min',
            ax=ax1,
            palette='husl',
            showmeans=True
        )
        ax1.set_xlabel('Patent Dataset', fontsize=11)
        ax1.set_ylabel('SIF Stability (minutes)', fontsize=11)
        ax1.set_title('SIF Stability Comparison (Box Plot)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

    # SGF 箱线图
    if len(df_sgf) > 0:
        sns.boxplot(
            data=df_sgf,
            x='dataset',
            y='SGF_class_min',
            ax=ax2,
            palette='husl',
            showmeans=True
        )
        ax2.set_xlabel('Patent Dataset', fontsize=11)
        ax2.set_ylabel('SGF Stability (minutes)', fontsize=11)
        ax2.set_title('SGF Stability Comparison (Box Plot)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'boxplot_comparison.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存箱线图对比: {output_path}")


def plot_dataset_statistics_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300
):
    """
    绘制数据集统计特征对比图

    Args:
        df: 合并的数据框
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    # 计算各数据集的统计特征
    stats_list = []

    for dataset_name, group in df.groupby('dataset'):
        stats = {
            'dataset': dataset_name,
            'total_samples': len(group),
            'monomer_rate': group['is_monomer'].mean() * 100,
            'dimer_rate': group['is_dimer'].mean() * 100,
            'cyclic_rate': group['is_cyclic'].mean() * 100,
            'disulfide_rate': group['has_disulfide_bond'].mean() * 100,
            'sif_valid_rate': (group['SIF_class_min'] != -1).mean() * 100,
            'sgf_valid_rate': (group['SGF_class_min'] != -1).mean() * 100,
        }

        # 计算标签均值和标准差（仅有效标签）
        sif_valid = group[group['SIF_class_min'] != -1]['SIF_class_min']
        sgf_valid = group[group['SGF_class_min'] != -1]['SGF_class_min']

        stats['sif_mean'] = sif_valid.mean() if len(sif_valid) > 0 else 0
        stats['sif_std'] = sif_valid.std() if len(sif_valid) > 0 else 0
        stats['sgf_mean'] = sgf_valid.mean() if len(sgf_valid) > 0 else 0
        stats['sgf_std'] = sgf_valid.std() if len(sgf_valid) > 0 else 0

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # === 绘制 4 个子图 ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Statistics Comparison Across Patents',
                 fontsize=16, fontweight='bold')

    # 1. 样本量对比
    ax1 = axes[0, 0]
    ax1.bar(stats_df['dataset'], stats_df['total_samples'],
            color=sns.color_palette("husl", len(stats_df)), alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Sample Size Comparison', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        ax1.text(i, row['total_samples'] + 10, str(row['total_samples']),
                ha='center', va='bottom', fontsize=9)

    # 2. 结构特征比率对比（堆叠条形图）
    ax2 = axes[0, 1]
    x = np.arange(len(stats_df))
    width = 0.2

    ax2.bar(x - width*1.5, stats_df['monomer_rate'], width, label='Monomer',
            color='steelblue', alpha=0.7, edgecolor='black')
    ax2.bar(x - width*0.5, stats_df['cyclic_rate'], width, label='Cyclic',
            color='coral', alpha=0.7, edgecolor='black')
    ax2.bar(x + width*0.5, stats_df['disulfide_rate'], width, label='Disulfide',
            color='lightgreen', alpha=0.7, edgecolor='black')

    ax2.set_ylabel('Rate (%)', fontsize=11)
    ax2.set_title('Structural Features Rate Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df['dataset'], rotation=45)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)

    # 3. SIF 标签均值对比（误差条）
    ax3 = axes[1, 0]
    ax3.bar(stats_df['dataset'], stats_df['sif_mean'],
            yerr=stats_df['sif_std'],
            color=sns.color_palette("husl", len(stats_df)),
            alpha=0.7, edgecolor='black', capsize=5)
    ax3.set_ylabel('SIF Stability (minutes)', fontsize=11)
    ax3.set_title('SIF Stability Mean ± Std', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)

    # 4. SGF 标签均值对比（误差条）
    ax4 = axes[1, 1]
    ax4.bar(stats_df['dataset'], stats_df['sgf_mean'],
            yerr=stats_df['sgf_std'],
            color=sns.color_palette("husl", len(stats_df)),
            alpha=0.7, edgecolor='black', capsize=5)
    ax4.set_ylabel('SGF Stability (minutes)', fontsize=11)
    ax4.set_title('SGF Stability Mean ± Std', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'dataset_statistics_comparison.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存数据集统计对比图: {output_path}")

    # 保存统计结果到 CSV
    stats_csv_path = output_dir / 'statistical_tests_results.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    logger.info(f"已保存统计结果 CSV: {stats_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='专利间可视化脚本 - Phase 2.2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--features_dir',
        type=Path,
        default=Path('outputs/features'),
        help='特征文件目录（NPZ 文件）'
    )
    parser.add_argument(
        '--processed_dir',
        type=Path,
        default=Path('data/processed'),
        help='处理后的 CSV 文件目录'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/figures/phase2/between_patents'),
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
    logger.info("开始专利间可视化分析 - Phase 2.2")
    logger.info("=" * 60)
    logger.info(f"特征目录: {args.features_dir}")
    logger.info(f"CSV 目录: {args.processed_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"图像分辨率: {args.dpi} DPI")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有特征数据（用于降维）
    logger.info("\n加载特征数据...")
    X_all, y_sif_all, y_sgf_all, source_labels = load_all_features(args.features_dir)

    if X_all is None:
        logger.error("特征数据加载失败，终止程序")
        return

    # 加载所有 CSV 数据（用于统计分析）
    logger.info("\n加载 CSV 数据...")
    df_all = load_all_csv_data(args.processed_dir)

    # 1. 降维可视化
    logger.info("\n生成降维可视化...")
    plot_dimensionality_reduction(X_all, y_sif_all, source_labels, args.output_dir, args.dpi)

    # 2. 标签分布对比
    logger.info("\n生成标签分布对比...")
    plot_label_distribution_comparison(df_all, args.output_dir, args.dpi)

    # 3. 数据集统计对比
    logger.info("\n生成数据集统计对比...")
    plot_dataset_statistics_comparison(df_all, args.output_dir, args.dpi)

    logger.info("\n" + "=" * 60)
    logger.info("专利间可视化分析完成！")
    logger.info(f"所有结果已保存到: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
