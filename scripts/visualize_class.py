#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别标签分布可视化脚本

从 data/cleaned/ 目录的 CSV 文件中提取 SIF/SGF 类别标签，并生成分布可视化图表。

用法：
    python scripts/visualize_class.py \\
        --input_dir data/cleaned/ \\
        --output_dir outputs/class_distribution/ \\
        --dpi 300 \\
        --format png
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 标签映射配置
LABEL_MAPPINGS = {
    'US9624268_cleaned': {
        'description': 'US9624268',
        'sif_labels': {
            1.0: '>360 min',
            2.0: '180-360 min',
            3.0: '120-180 min',
            4.0: '60-120 min',
            5.0: '<60 min'
        },
        'sgf_labels': {
            1.0: '>360 min',
            2.0: '180-360 min',
            3.0: '120-180 min',
            4.0: '60-120 min',
            5.0: '<60 min'
        }
    },
    'sif_sgf_second_cleaned': {
        'description': 'SIF/SGF Second',
        'sif_labels': {
            1: '<1 hour',
            2: '1-1.5 hours',
            3: '1.5-2 hours',
            4: '>2 hours'
        },
        'sgf_labels': {
            1: '<1 hour',
            2: '1-1.5 hours',
            3: '1.5-2 hours',
            4: '>2 hours'
        }
    }
}

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
            logging.FileHandler("visualize_class.log"),
        ],
    )


def get_label_mapping(dataset_name: str) -> Optional[Dict]:
    """
    根据数据集名称获取标签映射配置。

    Args:
        dataset_name (str): 数据集文件名（不包含后缀）。

    Returns:
        Optional[Dict]: 标签映射配置字典，若不存在则返回 None。
    """
    logger = logging.getLogger(__name__)
    
    # 完全匹配
    if dataset_name in LABEL_MAPPINGS:
        return LABEL_MAPPINGS[dataset_name]
    
    # 部分匹配（用于兼容带前缀的文件名）
    for key, mapping in LABEL_MAPPINGS.items():
        if key in dataset_name:
            logger.info(f"为 {dataset_name} 匹配到标签映射: {key}")
            return mapping
    
    logger.warning(f"未找到 {dataset_name} 的标签映射，将使用数字标签")
    return None


def load_csv_with_labels(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    加载 CSV 文件并提取 SIF/SGF 类别标签，允许单标签样本，缺失标签用 -1 表示。

    Args:
        csv_path (Path): CSV 文件路径。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
            - sif_classes: SIF 类别标签数组（缺失值用 -1 表示）
            - sgf_classes: SGF 类别标签数组（缺失值用 -1 表示）
            - ids: 样本 ID 数组
            - info_dict: 包含统计信息的字典

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺失必需列。
    """
    logger = logging.getLogger(__name__)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        required_columns = ['SIF_class', 'SGF_class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"CSV 文件缺失必需的列: {', '.join(missing_columns)}")
        
        total_rows = len(df)
        
        # 获取 id 列（如果存在）
        if 'id' in df.columns:
            ids = df['id'].astype(str).values
        else:
            ids = np.arange(total_rows, dtype=object)
        
        # 初始化标签数组
        sif_classes = []
        sgf_classes = []
        valid_ids = []
        
        # 遍历每一行，允许单标签样本
        for idx, row in df.iterrows():
            sif_class = row['SIF_class']
            sgf_class = row['SGF_class']
            
            # 检查两个标签是否都缺失 - 只有都缺失才跳过
            sif_is_na = bool(pd.isna(sif_class))
            sgf_is_na = bool(pd.isna(sgf_class))
            
            if sif_is_na and sgf_is_na:
                continue  # 跳过两个标签都缺失的行
            
            # 处理 SIF_class - 如果缺失则用 -1 表示
            sif_int = -1
            if not sif_is_na:
                sif_str = str(sif_class).strip()
                if sif_str == '' or sif_str == '----':
                    sif_int = -1
                else:
                    try:
                        sif_int = int(float(sif_str))
                    except (ValueError, TypeError):
                        sif_int = -1
            
            # 处理 SGF_class - 如果缺失则用 -1 表示
            sgf_int = -1
            if not sgf_is_na:
                sgf_str = str(sgf_class).strip()
                if sgf_str == '' or sgf_str == '----':
                    sgf_int = -1
                else:
                    try:
                        sgf_int = int(float(sgf_str))
                    except (ValueError, TypeError):
                        sgf_int = -1
            
            sif_classes.append(sif_int)
            sgf_classes.append(sgf_int)
            valid_ids.append(ids[idx])
        
        # 转换为 numpy 数组
        sif_classes = np.array(sif_classes, dtype=np.int32)
        sgf_classes = np.array(sgf_classes, dtype=np.int32)
        valid_ids = np.array(valid_ids, dtype=object)
        
        # 统计缺失值
        sif_missing_count = np.sum(sif_classes == -1)
        sgf_missing_count = np.sum(sgf_classes == -1)
        
        logger.info(
            f"成功加载 CSV 文件: {csv_path.name} "
            f"(总行数: {total_rows}, 有效行: {len(sif_classes)}, "
            f"SIF缺失: {sif_missing_count}, SGF缺失: {sgf_missing_count})"
        )
        
        info_dict = {
            'total_rows': total_rows,
            'valid_rows': len(sif_classes),
            'sif_missing': int(sif_missing_count),
            'sgf_missing': int(sgf_missing_count),
        }
        
        return sif_classes, sgf_classes, valid_ids, info_dict
    
    except Exception as e:
        logger.error(f"加载 CSV 文件出错: {e}")
        raise


def plot_class_distribution(
    sif_classes: np.ndarray,
    sgf_classes: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    label_mapping: Optional[Dict] = None,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制 SIF/SGF 类别分布条形图。

    Args:
        sif_classes (np.ndarray): SIF 类别标签数组。
        sgf_classes (np.ndarray): SGF 类别标签数组。
        dataset_name (str): 数据集名称。
        output_dir (Path): 输出目录。
        label_mapping (Optional[Dict]): 标签映射配置。
        dpi (int): 图像分辨率。默认 300。
        format (str): 输出格式。默认 "png"。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Class Distribution - {dataset_name}",
        fontsize=16, fontweight='bold'
    )
    
    # 获取标签映射
    sif_labels_map = None
    sgf_labels_map = None
    if label_mapping:
        sif_labels_map = label_mapping.get('sif_labels', {})
        sgf_labels_map = label_mapping.get('sgf_labels', {})
    
    # 绘制 SIF 分布 - 过滤掉 -1（缺失值）
    sif_valid = sif_classes[sif_classes != -1]
    sif_missing = np.sum(sif_classes == -1)
    sif_unique, sif_counts = np.unique(sif_valid.astype(int), return_counts=True)
    
    axes[0].bar(sif_unique, sif_counts, color='steelblue',
               edgecolor='black', alpha=0.7)
    
    # 构造 X 轴标签
    if sif_labels_map:
        sif_xticks = [f"{int(c)}\n{sif_labels_map.get(float(c), '')}" 
                     for c in sif_unique]
    else:
        sif_xticks = [str(int(c)) for c in sif_unique]
    
    axes[0].set_xticks(sif_unique)
    axes[0].set_xticklabels(sif_xticks, fontsize=9)
    axes[0].set_xlabel('SIF Class', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
    title_sif = f'SIF Stability Distribution (Missing: {sif_missing})'
    axes[0].set_title(title_sif, fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值和百分比
    total_sif = sif_counts.sum()
    for cls, v in zip(sif_unique, sif_counts):
        percentage = (v / total_sif) * 100
        axes[0].text(cls, v + total_sif * 0.01, f'{int(v)}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 绘制 SGF 分布 - 过滤掉 -1（缺失值）
    sgf_valid = sgf_classes[sgf_classes != -1]
    sgf_missing = np.sum(sgf_classes == -1)
    sgf_unique, sgf_counts = np.unique(sgf_valid.astype(int), return_counts=True)
    
    axes[1].bar(sgf_unique, sgf_counts, color='coral',
               edgecolor='black', alpha=0.7)
    
    # 构造 X 轴标签
    if sgf_labels_map:
        sgf_xticks = [f"{int(c)}\n{sgf_labels_map.get(float(c), '')}" 
                     for c in sgf_unique]
    else:
        sgf_xticks = [str(int(c)) for c in sgf_unique]
    
    axes[1].set_xticks(sgf_unique)
    axes[1].set_xticklabels(sgf_xticks, fontsize=9)
    axes[1].set_xlabel('SGF Class', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
    title_sgf = f'SGF Stability Distribution (Missing: {sgf_missing})'
    axes[1].set_title(title_sgf, fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值和百分比
    total_sgf = sgf_counts.sum()
    for cls, v in zip(sgf_unique, sgf_counts):
        percentage = (v / total_sgf) * 100
        axes[1].text(cls, v + total_sgf * 0.01, f'{int(v)}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f"{dataset_name}_class_distribution.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存类别分布图: {output_path}")
    plt.close()
    
    return output_path


def plot_joint_distribution(
    sif_classes: np.ndarray,
    sgf_classes: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    label_mapping: Optional[Dict] = None,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制 SIF 和 SGF 的联合分布热力图。

    Args:
        sif_classes (np.ndarray): SIF 类别标签数组。
        sgf_classes (np.ndarray): SGF 类别标签数组。
        dataset_name (str): 数据集名称。
        output_dir (Path): 输出目录。
        label_mapping (Optional[Dict]): 标签映射配置。
        dpi (int): 图像分辨率。默认 300。
        format (str): 输出格式。默认 "png"。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建交叉表，包含所有值（包括 -1）
    joint_dist = pd.crosstab(
        pd.Series(sif_classes, name='SIF'),
        pd.Series(sgf_classes, name='SGF')
    )
    
    # 调整坐标轴标签，包含时间范围信息和缺失类别
    label_mapping_config = label_mapping or {}
    sif_labels_map = label_mapping_config.get('sif_labels', {})
    sgf_labels_map = label_mapping_config.get('sgf_labels', {})
    
    # 创建标签字典，将 -1 映射为 "Missing"
    index_labels = []
    for idx in joint_dist.index:
        if idx == -1:
            index_labels.append("Missing")
        elif sif_labels_map:
            index_labels.append(f"{int(idx)}\n{sif_labels_map.get(float(idx), '')}")
        else:
            index_labels.append(str(int(idx)))
    
    column_labels = []
    for col in joint_dist.columns:
        if col == -1:
            column_labels.append("Missing")
        elif sgf_labels_map:
            column_labels.append(f"{int(col)}\n{sgf_labels_map.get(float(col), '')}")
        else:
            column_labels.append(str(int(col)))
    
    # 重新标记行和列
    joint_dist.index = index_labels
    joint_dist.columns = column_labels
    
    # 绘制热力图
    sns.heatmap(joint_dist, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Count'})
    
    # 设置 Y 轴标签（SIF）
    ax.set_yticklabels(index_labels, rotation=0, fontsize=9)
    
    # 设置 X 轴标签（SGF）
    ax.set_xticklabels(column_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel('SGF Stability Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('SIF Stability Class', fontsize=12, fontweight='bold')
    
    # 添加数据集描述信息
    desc = label_mapping_config.get('description', dataset_name)
    ax.set_title(
        f"Joint Distribution of SIF and SGF - {dataset_name}\n({desc})",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    output_path = output_dir / f"{dataset_name}_joint_distribution.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存联合分布热力图: {output_path}")
    plt.close()
    
    return output_path


def generate_statistics_summary(
    sif_classes: np.ndarray,
    sgf_classes: np.ndarray,
    ids: np.ndarray,
    dataset_name: str,
    csv_path: Path,
    label_mapping: Optional[Dict] = None
) -> Dict:
    """
    生成数据集的统计摘要信息。

    包含缺失标签的统计信息（-1 值）及其对应的样本 ID。

    Args:
        sif_classes (np.ndarray): SIF 类别标签数组（包含 -1 表示缺失）。
        sgf_classes (np.ndarray): SGF 类别标签数组（包含 -1 表示缺失）。
        ids (np.ndarray): 样本 ID 数组。
        dataset_name (str): 数据集名称。
        csv_path (Path): CSV 文件路径。
        label_mapping (Optional[Dict]): 标签映射配置。

    Returns:
        Dict: 包含统计摘要信息的字典。
    """
    logger = logging.getLogger(__name__)
    
    # 获取标签映射
    sif_labels_map = {}
    sgf_labels_map = {}
    if label_mapping:
        sif_labels_map = label_mapping.get('sif_labels', {})
        sgf_labels_map = label_mapping.get('sgf_labels', {})
    
    # 计算缺失标签统计信息
    sif_missing_mask = sif_classes == -1
    sgf_missing_mask = sgf_classes == -1
    sif_missing_count = np.sum(sif_missing_mask)
    sgf_missing_count = np.sum(sgf_missing_mask)
    
    # 获取缺失标签的 ID
    sif_missing_ids = ids[sif_missing_mask].tolist() if sif_missing_count > 0 else []
    sgf_missing_ids = ids[sgf_missing_mask].tolist() if sgf_missing_count > 0 else []
    
    # 计算 SIF 分布（排除 -1）
    sif_valid = sif_classes[sif_classes != -1]
    if len(sif_valid) > 0:
        sif_unique, sif_counts = np.unique(sif_valid, return_counts=True)
        sif_distribution = {}
        total_sif = len(sif_valid)
        for cls, count in zip(sif_unique, sif_counts):
            sif_distribution[str(int(cls))] = {
                'count': int(count),
                'percentage': round(count / total_sif * 100, 2),
                'label': sif_labels_map.get(float(cls), f'Class {int(cls)}')
            }
    else:
        sif_distribution = {}
    
    # 计算 SGF 分布（排除 -1）
    sgf_valid = sgf_classes[sgf_classes != -1]
    if len(sgf_valid) > 0:
        sgf_unique, sgf_counts = np.unique(sgf_valid, return_counts=True)
        sgf_distribution = {}
        total_sgf = len(sgf_valid)
        for cls, count in zip(sgf_unique, sgf_counts):
            sgf_distribution[str(int(cls))] = {
                'count': int(count),
                'percentage': round(count / total_sgf * 100, 2),
                'label': sgf_labels_map.get(float(cls), f'Class {int(cls)}')
            }
    else:
        sgf_distribution = {}
    
    # 读取原始 CSV 获取总行数
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
    except Exception as e:
        logger.warning(f"读取 CSV 信息失败: {e}")
        total_rows = len(sif_classes)
    
    summary = {
        'dataset_name': dataset_name,
        'csv_file': str(csv_path.name),
        'total_rows': total_rows,
        'valid_rows': len(sif_classes),
        'valid_rows_sif': len(sif_valid),
        'valid_rows_sgf': len(sgf_valid),
        'missing_values': {
            'sif': {
                'count': int(sif_missing_count),
                'ids': sif_missing_ids
            },
            'sgf': {
                'count': int(sgf_missing_count),
                'ids': sgf_missing_ids
            }
        },
        'sif_distribution': sif_distribution,
        'sgf_distribution': sgf_distribution
    }
    
    if label_mapping:
        summary['mapping_description'] = label_mapping.get('description', '')
    
    logger.info(f"生成 {dataset_name} 的统计摘要")
    return summary


def process_single_csv(
    csv_path: Path,
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Dict:
    """
    处理单个 CSV 文件并生成可视化和统计摘要。

    Args:
        csv_path (Path): 输入 CSV 文件路径。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。默认 300。
        format (str): 输出格式。默认 "png"。

    Returns:
        Dict: 包含处理结果的字典。
    """
    logger = logging.getLogger(__name__)
    dataset_name = csv_path.stem
    
    try:
        # 加载数据
        sif_classes, sgf_classes, ids, info_dict = load_csv_with_labels(csv_path)
        
        # 获取标签映射
        label_mapping = get_label_mapping(dataset_name)
        
        logger.info(f"开始处理 {dataset_name}...")
        
        # 生成条形图
        plot_class_distribution(
            sif_classes, sgf_classes, dataset_name, output_dir,
            label_mapping, dpi, format
        )
        
        # 生成联合分布热力图
        plot_joint_distribution(
            sif_classes, sgf_classes, dataset_name, output_dir,
            label_mapping, dpi, format
        )
        
        # 生成统计摘要
        summary = generate_statistics_summary(
            sif_classes, sgf_classes, ids, dataset_name, csv_path, label_mapping
        )
        
        logger.info(f"完成 {dataset_name} 的处理")
        
        return {
            'dataset': dataset_name,
            'success': True,
            'summary': summary,
            'missing_values': summary.get('missing_values', {'sif': {'count': 0, 'ids': []}, 'sgf': {'count': 0, 'ids': []}})
        }
    
    except Exception as e:
        logger.error(f"处理文件 {csv_path.name} 时出错: {e}")
        return {
            'dataset': dataset_name,
            'success': False,
            'error': str(e)
        }


def get_csv_files(directory: Path) -> List[Path]:
    """
    获取目录下所有 CSV 文件。

    Args:
        directory (Path): 输入目录。

    Returns:
        List[Path]: CSV 文件路径列表。
    """
    logger = logging.getLogger(__name__)
    
    if not directory.exists():
        logger.error(f"目录不存在: {directory}")
        return []
    
    csv_files = sorted(directory.glob("*.csv"))
    logger.info(f"在 {directory} 中找到 {len(csv_files)} 个 CSV 文件")
    
    return csv_files


def main() -> int:
    """
    主函数：批量处理类别分布可视化。

    Returns:
        int: 程序退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(
        description="从 CSV 文件生成 SIF/SGF 类别分布可视化图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/visualize_class.py "
            "--input_dir data/cleaned/ --output_dir outputs/class_distribution/\n"
        ),
    )
    
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/cleaned"),
        help="输入 CSV 文件所在的目录（默认: data/cleaned）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/class_distribution"),
        help="输出可视化图表的目录（默认: outputs/class_distribution）",
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
    logger.info("开始 SIF/SGF 类别分布可视化")
    logger.info("=" * 70)
    logger.info(f"输入目录: {args.input_dir.resolve()}")
    logger.info(f"输出目录: {args.output_dir.resolve()}")
    logger.info(f"图像分辨率: {args.dpi} DPI")
    logger.info(f"输出格式: {args.format}")
    
    # 验证输入目录
    if not args.input_dir.exists():
        logger.error(f"输入目录不存在: {args.input_dir}")
        return 2
    
    if not args.input_dir.is_dir():
        logger.error(f"输入路径不是目录: {args.input_dir}")
        return 2
    
    # 创建输出目录
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已创建: {args.output_dir.resolve()}")
    except OSError as e:
        logger.error(f"创建输出目录失败: {e}")
        return 2
    
    # 获取所有 CSV 文件
    csv_files = get_csv_files(args.input_dir)
    
    if len(csv_files) == 0:
        logger.warning(f"在 {args.input_dir} 中未找到 CSV 文件")
        return 1
    
    logger.info(f"找到 {len(csv_files)} 个 CSV 文件待处理")
    
    # 批量处理 CSV 文件
    all_results = []
    successful_files = 0
    
    for csv_path in csv_files:
        result = process_single_csv(
            csv_path,
            args.output_dir,
            dpi=args.dpi,
            format=args.format
        )
        all_results.append(result)
        
        if result['success']:
            successful_files += 1
    
    # 保存结果摘要
    summary_path = args.output_dir / "class_distribution_summary.json"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"保存可视化摘要: {summary_path}")
    except Exception as e:
        logger.error(f"保存摘要失败: {e}")
    
    # 输出最终摘要
    logger.info("=" * 70)
    logger.info("类别分布可视化生成完成")
    logger.info("=" * 70)
    logger.info(f"处理文件数: {successful_files}/{len(csv_files)}")
    logger.info(f"输出目录: {args.output_dir.resolve()}")
    
    if successful_files == 0:
        logger.error("没有文件成功处理")
        return 1
    
    logger.info("类别分布可视化完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())

