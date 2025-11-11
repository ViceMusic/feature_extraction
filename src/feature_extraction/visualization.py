"""
数据可视化模块

提供 DataVisualizer 类，用于 SIF/SGF 肽类稳定性数据的探索性数据分析（EDA）可视化。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """
    数据可视化工具类。

    用于从 NPZ 文件中加载特征数据，并生成多种探索性数据分析（EDA）图表。

    Attributes:
        X (np.ndarray): 特征矩阵，shape: (n_samples, n_features)。
        y_sif (np.ndarray): SIF 稳定性标签，shape: (n_samples,)。
        y_sgf (np.ndarray): SGF 稳定性标签，shape: (n_samples,)。
        feature_names (np.ndarray): 特征名称列表。
        dataset_name (str): 数据集名称。
        df_data (pd.DataFrame): 包含特征的数据框。
        dpi (int): 图像分辨率。
    """

    def __init__(
        self,
        X: np.ndarray,
        y_sif: np.ndarray,
        y_sgf: np.ndarray,
        feature_names: np.ndarray,
        dataset_name: str = "Dataset",
        dpi: int = 300,
    ) -> None:
        """
        初始化数据可视化器。

        Args:
            X (np.ndarray): 特征矩阵。
            y_sif (np.ndarray): SIF 标签。
            y_sgf (np.ndarray): SGF 标签。
            feature_names (np.ndarray): 特征名称。
            dataset_name (str): 数据集名称。默认 "Dataset"。
            dpi (int): 图像分辨率。默认 300。
        """
        self.X = X
        self.y_sif = y_sif
        self.y_sgf = y_sgf
        self.feature_names = feature_names
        self.dataset_name = dataset_name
        self.dpi = dpi

        # 创建数据框
        self.df_data = pd.DataFrame(
            self.X, columns=self.feature_names
        )
        self.df_data['SIF_class'] = self.y_sif
        self.df_data['SGF_class'] = self.y_sgf

        logger.info(
            f"初始化 DataVisualizer: {dataset_name}, "
            f"样本数={len(X)}, 特征数={len(feature_names)}"
        )

    def _get_feature_by_name(self, name: str) -> Optional[np.ndarray]:
        """
        按名称获取特征向量。

        Args:
            name (str): 特征名称。

        Returns:
            Optional[np.ndarray]: 特征向量，若不存在则返回 None。
        """
        if name in self.feature_names:
            idx = list(self.feature_names).index(name)
            return self.X[:, idx]
        return None

    def plot_feature_distributions(
        self, output_dir: Path, format: str = "png"
    ) -> Path:
        """
        绘制基础物理化学特征分布图。

        生成 4 个直方图：Molecular Weight、LogP、HBA、HBD。

        Args:
            output_dir (Path): 输出目录。
            format (str): 输出格式。默认 "png"。

        Returns:
            Path: 保存的图表路径。
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Basic Physicochemical Features - {self.dataset_name}",
            fontsize=16, fontweight='bold'
        )

        features_to_plot = [
            ('PC_MolWt', 'Molecular Weight (g/mol)'),
            ('PC_LogP', 'LogP (Lipophilicity)'),
            ('PC_HBA', 'Hydrogen Bond Acceptors'),
            ('PC_HBD', 'Hydrogen Bond Donors'),
        ]

        axes = axes.flatten()

        for ax, (feature_name, label) in zip(axes, features_to_plot):
            feature_data = self._get_feature_by_name(feature_name)

            if feature_data is None:
                logger.warning(f"特征 {feature_name} 不存在")
                ax.text(0.5, 0.5, f"Feature {feature_name} not found",
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # 绘制直方图和核密度估计
            ax.hist(feature_data, bins=30, alpha=0.6, color='skyblue',
                   edgecolor='black', density=True, label='Histogram')
            
            # 核密度估计曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(feature_data)
            x_range = np.linspace(feature_data.min(), feature_data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            # 添加均值和中位数线
            mean_val = np.mean(feature_data)
            median_val = np.median(feature_data)
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {median_val:.2f}')

            # 统计信息
            std_val = np.std(feature_data)
            skew_val = (
                3 * (mean_val - median_val) / std_val if std_val > 0 else 0
            )

            stats_text = (
                f"N={len(feature_data)}\n"
                f"μ={mean_val:.2f}\n"
                f"σ={std_val:.2f}\n"
                f"Skew={skew_val:.2f}"
            )
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{self.dataset_name}_feature_distributions.{format}"
        plt.savefig(output_path, dpi=self.dpi, format=format, bbox_inches='tight')
        logger.info(f"保存特征分布图: {output_path}")
        plt.close()

        return output_path

    def plot_label_distribution(
        self, output_dir: Path, format: str = "png"
    ) -> List[Path]:
        """
        绘制稳定性标签分布图。

        生成两个图：SIF/SGF 标签计数柱状图和联合分布热力图。

        Args:
            output_dir (Path): 输出目录。
            format (str): 输出格式。默认 "png"。

        Returns:
            List[Path]: 保存的图表路径列表。
        """
        output_paths = []

        # 1. 标签计数柱状图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Stability Label Distribution - {self.dataset_name}",
            fontsize=14, fontweight='bold'
        )

        # SIF 分布 - 使用 unique 获取实际存在的标签
        sif_unique, sif_counts = np.unique(self.y_sif.astype(int), return_counts=True)
        axes[0].bar(sif_unique, sif_counts, color='steelblue',
                   edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('SIF Stability Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('SIF Distribution')
        axes[0].set_xticks(sif_unique)
        for cls, v in zip(sif_unique, sif_counts):
            axes[0].text(cls, v + 5, str(v), ha='center', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # SGF 分布 - 使用 unique 获取实际存在的标签
        sgf_unique, sgf_counts = np.unique(self.y_sgf.astype(int), return_counts=True)
        axes[1].bar(sgf_unique, sgf_counts, color='coral',
                   edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('SGF Stability Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('SGF Distribution')
        axes[1].set_xticks(sgf_unique)
        for cls, v in zip(sgf_unique, sgf_counts):
            axes[1].text(cls, v + 5, str(v), ha='center', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path1 = (
            output_dir / f"{self.dataset_name}_label_distribution.{format}"
        )
        plt.savefig(output_path1, dpi=self.dpi, format=format, bbox_inches='tight')
        logger.info(f"保存标签分布图: {output_path1}")
        plt.close()
        output_paths.append(output_path1)

        # 2. 联合分布热力图
        fig, ax = plt.subplots(figsize=(8, 6))

        # 创建交叉表
        joint_dist = pd.crosstab(self.y_sif, self.y_sgf)
        sns.heatmap(joint_dist, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('SGF Stability Class')
        ax.set_ylabel('SIF Stability Class')
        ax.set_title(
            f"Joint Distribution of SIF and SGF - {self.dataset_name}"
        )

        plt.tight_layout()
        output_path2 = (
            output_dir / f"{self.dataset_name}_label_joint_distribution.{format}"
        )
        plt.savefig(output_path2, dpi=self.dpi, format=format, bbox_inches='tight')
        logger.info(f"保存联合分布图: {output_path2}")
        plt.close()
        output_paths.append(output_path2)

        return output_paths

    def plot_grouped_boxplots(
        self, output_dir: Path, format: str = "png"
    ) -> Path:
        """
        绘制按稳定性分组的箱线图。

        显示分子量和 LogP 与 SIF/SGF 稳定性等级的关系。

        Args:
            output_dir (Path): 输出目录。
            format (str): 输出格式。默认 "png"。

        Returns:
            Path: 保存的图表路径。
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Features vs Stability Classes - {self.dataset_name}",
            fontsize=16, fontweight='bold'
        )

        mw_data = self._get_feature_by_name('PC_MolWt')
        logp_data = self._get_feature_by_name('PC_LogP')

        plot_specs = [
            (0, 0, 'PC_MolWt', mw_data, 'SIF_class', 'Molecular Weight vs SIF'),
            (0, 1, 'PC_LogP', logp_data, 'SIF_class', 'LogP vs SIF'),
            (1, 0, 'PC_MolWt', mw_data, 'SGF_class', 'Molecular Weight vs SGF'),
            (1, 1, 'PC_LogP', logp_data, 'SGF_class', 'LogP vs SGF'),
        ]

        for row, col, feat_name, feat_data, label_name, title in plot_specs:
            if feat_data is None:
                logger.warning(f"特征 {feat_name} 不存在")
                continue

            ax = axes[row, col]
            df_plot = pd.DataFrame({
                'Value': feat_data,
                'Class': self.df_data[label_name]
            })

            sns.boxplot(data=df_plot, x='Class', y='Value', ax=ax,
                       palette='Set2')
            ax.set_title(title)
            ax.set_ylabel(feat_name)
            ax.set_xlabel('Stability Class')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = (
            output_dir / f"{self.dataset_name}_grouped_boxplots.{format}"
        )
        plt.savefig(output_path, dpi=self.dpi, format=format, bbox_inches='tight')
        logger.info(f"保存分组箱线图: {output_path}")
        plt.close()

        return output_path

    def plot_correlation_heatmap(
        self, output_dir: Path, format: str = "png"
    ) -> Path:
        """
        绘制关键特征相关性热力图。

        显示物理化学特征间的 Pearson 相关系数和特征与稳定性标签的 Spearman 相关系数。

        Args:
            output_dir (Path): 输出目录。
            format (str): 输出格式。默认 "png"。

        Returns:
            Path: 保存的图表路径。
        """
        # 选择关键特征
        key_features = [
            'PC_MolWt', 'PC_LogP', 'PC_HBA', 'PC_HBD', 'PC_TPSA',
            'PC_Rings', 'PC_RigidityProxy'
        ]

        # 获取可用的特征
        available_features = [
            f for f in key_features if f in self.feature_names
        ]

        if len(available_features) < 2:
            logger.warning("可用的关键特征过少，跳过相关性热力图")
            return None

        df_features = self.df_data[available_features].copy()
        df_features['SIF_class'] = self.y_sif
        df_features['SGF_class'] = self.y_sgf

        # 计算相关系数矩阵
        # 特征间使用 Pearson，特征与标签使用 Spearman
        corr_matrix = df_features.corr(method='pearson')

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                   vmin=-1, vmax=1)
        ax.set_title(
            f"Correlation Matrix - {self.dataset_name}\n"
            "(Pearson for features, Spearman used for labels)"
        )

        plt.tight_layout()
        output_path = (
            output_dir / f"{self.dataset_name}_correlation_heatmap.{format}"
        )
        plt.savefig(output_path, dpi=self.dpi, format=format, bbox_inches='tight')
        logger.info(f"保存相关性热力图: {output_path}")
        plt.close()

        return output_path

    def plot_scatter_matrix(
        self, output_dir: Path, format: str = "png"
    ) -> Path:
        """
        绘制散点图矩阵。

        显示关键特征间的二维关系，用 SIF 稳定性等级着色。

        Args:
            output_dir (Path): 输出目录。
            format (str): 输出格式。默认 "png"。

        Returns:
            Path: 保存的图表路径。
        """
        # 选择特征用于散点矩阵
        features_to_plot = [
            'PC_MolWt', 'PC_LogP', 'PC_HBA', 'PC_HBD'
        ]

        available_features = [
            f for f in features_to_plot if f in self.feature_names
        ]

        if len(available_features) < 2:
            logger.warning("可用特征过少，跳过散点矩阵")
            return None

        df_scatter = self.df_data[available_features].copy()
        df_scatter['SIF_class'] = self.y_sif

        # 创建散点矩阵
        from pandas.plotting import scatter_matrix

        fig = plt.figure(figsize=(12, 12))
        scatter_matrix(
            df_scatter,
            alpha=0.6,
            figsize=(12, 12),
            diagonal='hist',
            c=df_scatter['SIF_class'],
            cmap='viridis'
        )

        fig.suptitle(
            f"Scatter Matrix (colored by SIF class) - {self.dataset_name}",
            fontsize=14, fontweight='bold', y=0.995
        )

        plt.tight_layout()
        output_path = (
            output_dir / f"{self.dataset_name}_scatter_matrix.{format}"
        )
        plt.savefig(output_path, dpi=self.dpi, format=format, bbox_inches='tight')
        logger.info(f"保存散点矩阵: {output_path}")
        plt.close()

        return output_path

    def generate_summary_statistics(self) -> Dict[str, any]:
        """
        生成数据集的摘要统计信息。

        Returns:
            Dict[str, any]: 包含摘要统计信息的字典。
        """
        key_features = [
            'PC_MolWt', 'PC_LogP', 'PC_HBA', 'PC_HBD'
        ]

        summary = {
            'dataset_name': self.dataset_name,
            'n_samples': len(self.X),
            'n_features': len(self.feature_names),
            'sif_label_distribution': dict(zip(*np.unique(
                self.y_sif, return_counts=True
            ))),
            'sgf_label_distribution': dict(zip(*np.unique(
                self.y_sgf, return_counts=True
            ))),
            'feature_statistics': {}
        }

        for feat in key_features:
            if feat in self.feature_names:
                data = self._get_feature_by_name(feat)
                summary['feature_statistics'][feat] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'median': float(np.median(data)),
                }

        return summary

