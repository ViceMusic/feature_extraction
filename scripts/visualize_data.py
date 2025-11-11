#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可视化脚本

从 NPZ 特征文件加载数据，生成探索性数据分析（EDA）可视化图表。

用法：
    python scripts/visualize_data.py \\
        --input_dir outputs/features/ \\
        --output_dir outputs/figures/ \\
        --dpi 300 \\
        --format png
"""

import argparse
import logging
import json
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_extraction.visualization import DataVisualizer


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
            logging.FileHandler("visualization.log"),
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


def get_npz_files(directory: Path) -> list:
    """
    获取目录下所有 NPZ 文件。

    Args:
        directory (Path): 输入目录。

    Returns:
        list: NPZ 文件路径列表。
    """
    logger = logging.getLogger(__name__)

    if not directory.exists():
        logger.error(f"目录不存在: {directory}")
        return []

    npz_files = sorted(directory.glob("*.npz"))
    logger.info(f"在 {directory} 中找到 {len(npz_files)} 个 NPZ 文件")

    return npz_files


def generate_visualizations_for_file(
    npz_path: Path,
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> dict:
    """
    为单个 NPZ 文件生成所有可视化图表。

    Args:
        npz_path (Path): 输入 NPZ 文件路径。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。默认 300。
        format (str): 输出格式。默认 "png"。

    Returns:
        dict: 包含生成信息的字典。
    """
    logger = logging.getLogger(__name__)

    try:
        # 加载 NPZ 文件
        X, y_sif, y_sgf, feature_names = load_npz_file(npz_path)

        # 获取数据集名称（不含后缀）
        dataset_name = npz_path.stem

        # 如果是 sif_sgf_second（包括带 _cleaned 后缀），将标签中的 5 转为 4
        if 'sif_sgf_second' in dataset_name:
            logger.info("检测到 sif_sgf_second 数据集，将标签中的 5 转为 4...")
            y_sif_before = np.unique(y_sif)
            y_sgf_before = np.unique(y_sgf)
            
            y_sif = np.where(y_sif == 5, 4, y_sif)
            y_sgf = np.where(y_sgf == 5, 4, y_sgf)
            
            logger.info(f"转换前 - y_sif: {y_sif_before}, y_sgf: {y_sgf_before}")
            logger.info(f"转换后 - y_sif: {np.unique(y_sif)}, y_sgf: {np.unique(y_sgf)}")

        # 创建可视化器
        visualizer = DataVisualizer(
            X=X,
            y_sif=y_sif,
            y_sgf=y_sgf,
            feature_names=feature_names,
            dataset_name=dataset_name,
            dpi=dpi
        )

        # 生成所有图表
        logger.info(f"开始生成 {dataset_name} 的可视化图表...")

        generated_files = []

        # 1. 特征分布图
        try:
            path = visualizer.plot_feature_distributions(output_dir, format)
            generated_files.append(str(path))
        except Exception as e:
            logger.warning(f"生成特征分布图失败: {e}")

        # 2. 标签分布图
        try:
            paths = visualizer.plot_label_distribution(output_dir, format)
            generated_files.extend([str(p) for p in paths])
        except Exception as e:
            logger.warning(f"生成标签分布图失败: {e}")

        # 3. 分组箱线图
        try:
            path = visualizer.plot_grouped_boxplots(output_dir, format)
            generated_files.append(str(path))
        except Exception as e:
            logger.warning(f"生成分组箱线图失败: {e}")

        # 4. 相关性热力图
        try:
            path = visualizer.plot_correlation_heatmap(output_dir, format)
            if path is not None:
                generated_files.append(str(path))
        except Exception as e:
            logger.warning(f"生成相关性热力图失败: {e}")

        # 5. 散点矩阵
        try:
            path = visualizer.plot_scatter_matrix(output_dir, format)
            if path is not None:
                generated_files.append(str(path))
        except Exception as e:
            logger.warning(f"生成散点矩阵失败: {e}")

        # 生成摘要统计
        summary = visualizer.generate_summary_statistics()

        logger.info(
            f"完成 {dataset_name} 的可视化生成，共 {len(generated_files)} 个图表"
        )

        return {
            'dataset': dataset_name,
            'success': True,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'n_figures': len(generated_files),
            'figures': generated_files,
            'summary': summary
        }

    except Exception as e:
        logger.error(f"处理文件 {npz_path.name} 时出错: {e}")
        return {
            'dataset': npz_path.stem,
            'success': False,
            'error': str(e)
        }


def main() -> int:
    """
    主函数：批量生成可视化图表。

    Returns:
        int: 程序退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(
        description="从特征 NPZ 文件生成探索性数据分析（EDA）可视化图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/visualize_data.py "
            "--input_dir outputs/features/ --output_dir outputs/figures/\n"
        ),
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/features"),
        help="输入 NPZ 文件所在的目录（默认: outputs/features）",
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
    logger.info("开始 SIF/SGF 数据可视化")
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

    # 获取所有 NPZ 文件
    npz_files = get_npz_files(args.input_dir)

    if len(npz_files) == 0:
        logger.warning(f"在 {args.input_dir} 中未找到 NPZ 文件")
        return 1

    logger.info(f"找到 {len(npz_files)} 个 NPZ 文件待处理")

    # 批量处理 NPZ 文件
    all_results = []
    successful_files = 0
    total_figures = 0

    for npz_path in npz_files:
        result = generate_visualizations_for_file(
            npz_path,
            args.output_dir,
            dpi=args.dpi,
            format=args.format
        )
        all_results.append(result)

        if result['success']:
            successful_files += 1
            total_figures += result['n_figures']

    # 保存结果摘要
    summary_path = args.output_dir / "visualization_summary.json"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"保存可视化摘要: {summary_path}")
    except Exception as e:
        logger.error(f"保存摘要失败: {e}")

    # 输出最终摘要
    logger.info("=" * 70)
    logger.info("可视化生成完成")
    logger.info("=" * 70)
    logger.info(f"处理文件数: {successful_files}/{len(npz_files)}")
    logger.info(f"生成图表数: {total_figures}")
    logger.info(f"输出目录: {args.output_dir.resolve()}")

    if successful_files == 0:
        logger.error("没有文件成功处理")
        return 1

    logger.info("数据可视化完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())

