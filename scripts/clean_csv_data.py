"""
CSV 数据清理脚本。

本脚本用于清理两个 CSV 文件（US9624268.csv 和 sif_sgf_second.csv）：
- 将星号标记转换为数字
- 将 '---' 占位符替换为空值
- 删除所有空列
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_star_to_number(value: str) -> Optional[int]:
    """
    将星号标记转换为数字。
    
    将形如 '*', '**', '***' 等的星号字符串转换为对应的数字。
    如果输入不是星号字符串，则返回 NaN。
    
    Args:
        value (str): 输入值，可能是星号字符串。
    
    Returns:
        Optional[int]: 星号的数量，如果输入不是星号则返回 NaN。
    
    Examples:
        >>> convert_star_to_number('*')
        1.0
        >>> convert_star_to_number('*****')
        5.0
        >>> convert_star_to_number('1')
        nan
    """
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # 检查是否全由星号组成
    if value_str and all(c == '*' for c in value_str):
        return float(len(value_str))
    
    # 如果已经是数字，返回数字
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return pd.NA


def replace_dash_with_nan(value: str) -> Optional[str]:
    """
    将 '---' 占位符替换为空值。
    
    如果值为 '---'，则返回 NaN；否则返回原值。
    
    Args:
        value (str): 输入值。
    
    Returns:
        Optional[str]: 如果是 '---' 则返回 NaN，否则返回原值。
    """
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    if value_str == '---':
        return pd.NA
    
    return value


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除所有空列。
    
    删除其中所有值都是 NaN/空的列。
    
    Args:
        df (pd.DataFrame): 输入的 DataFrame。
    
    Returns:
        pd.DataFrame: 删除空列后的 DataFrame。
    """
    # 计算每列的非 NaN 值数量
    non_null_counts = df.notna().sum()
    
    # 找到所有非空列
    non_empty_columns = non_null_counts[non_null_counts > 0].index.tolist()
    
    logger.info(f"删除了 {len(df.columns) - len(non_empty_columns)} 个空列，保留了 {len(non_empty_columns)} 个非空列")
    
    return df[non_empty_columns]


def clean_us9624268(input_path: Path, output_path: Path) -> None:
    """
    清理 US9624268.csv 文件。
    
    处理流程：
    1. 将 SIF_class 和 SGF_class 列中的星号标记转换为数字
    2. 将 '---' 替换为空值
    3. 删除所有空列
    
    Args:
        input_path (Path): 输入文件路径。
        output_path (Path): 输出文件路径。
    
    Raises:
        FileNotFoundError: 如果输入文件不存在。
    """
    logger.info(f"开始清理 US9624268.csv：{input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    
    # 读取 CSV 文件
    df = pd.read_csv(input_path)
    logger.info(f"读取了 {len(df)} 行，{len(df.columns)} 列")
    
    # 将 '---' 替换为 NaN
    df = df.map(replace_dash_with_nan)
    logger.info("已将 '---' 替换为空值")
    
    # 转换星号标记为数字（处理 SIF_class 和 SGF_class 列）
    for col in ['SIF_class', 'SGF_class']:
        if col in df.columns:
            df[col] = df[col].apply(convert_star_to_number)
            logger.info(f"已转换 {col} 列的星号标记为数字")
    
    # 删除空列
    df = remove_empty_columns(df)
    
    # 保存清理后的文件
    df.to_csv(output_path, index=False)
    logger.info(f"已保存清理后的文件：{output_path}，包含 {len(df)} 行，{len(df.columns)} 列")


def clean_sif_sgf_second(input_path: Path, output_path: Path) -> None:
    """
    清理 sif_sgf_second.csv 文件。
    
    处理流程：
    1. 将 '---' 替换为空值
    2. 删除所有空列
    
    Args:
        input_path (Path): 输入文件路径。
        output_path (Path): 输出文件路径。
    
    Raises:
        FileNotFoundError: 如果输入文件不存在。
    """
    logger.info(f"开始清理 sif_sgf_second.csv：{input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    
    # 读取 CSV 文件
    df = pd.read_csv(input_path)
    logger.info(f"读取了 {len(df)} 行，{len(df.columns)} 列")
    
    # 将 '---' 替换为 NaN
    df = df.map(replace_dash_with_nan)
    logger.info("已将 '---' 替换为空值")
    
    # 删除空列
    df = remove_empty_columns(df)
    
    # 保存清理后的文件
    df.to_csv(output_path, index=False)
    logger.info(f"已保存清理后的文件：{output_path}，包含 {len(df)} 行，{len(df.columns)} 列")


def main() -> None:
    """
    主函数，执行两个 CSV 文件的清理操作。
    """
    # 定义文件路径
    base_dir = Path(__file__).parent
    
    us9624268_input = base_dir / "US9624268.csv"
    us9624268_output = base_dir / "US9624268_cleaned.csv"
    
    sif_sgf_second_input = base_dir / "sif_sgf_second.csv"
    sif_sgf_second_output = base_dir / "sif_sgf_second_cleaned.csv"
    
    try:
        # 清理 US9624268.csv
        clean_us9624268(us9624268_input, us9624268_output)
        logger.info("✓ US9624268.csv 清理完成")
        
        # 清理 sif_sgf_second.csv
        clean_sif_sgf_second(sif_sgf_second_input, sif_sgf_second_output)
        logger.info("✓ sif_sgf_second.csv 清理完成")
        
        logger.info("所有文件清理完成！")
        
    except FileNotFoundError as e:
        logger.error(f"文件错误：{e}")
        raise
    except Exception as e:
        logger.error(f"处理过程中出现错误：{e}")
        raise


if __name__ == "__main__":
    main()

