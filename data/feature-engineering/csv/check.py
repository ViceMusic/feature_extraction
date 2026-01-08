import pandas as pd
from pathlib import Path

def check_smiles_overlap(file1_path: str, file2_path: str, smiles_col: str = "SMILES"):
    """
    检查两个 CSV 文件之间 SMILES 字段的重叠情况
    """
    p1 = Path(file1_path)
    p2 = Path(file2_path)
    
    # 1. 加载数据
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    
    # 2. 提取唯一的 SMILES 集合 (去除各自内部的重复)
    set1 = set(df1[smiles_col].dropna().unique())
    set2 = set(df2[smiles_col].dropna().unique())
    
    # 3. 计算交集
    intersection = set1.intersection(set2)
    
    # 4. 计算统计数据
    total_unique_all = len(set1.union(set2))
    overlap_count = len(intersection)
    
    print("="*50)
    print(f"数据重叠检查报告")
    print("="*50)
    print(f"文件 A: {p1.name}")
    print(f"  - 总行数: {len(df1)}")
    print(f"  - 唯一 SMILES 数: {len(set1)}")
    print(f"-"*30)
    print(f"文件 B: {p2.name}")
    print(f"  - 总行数: {len(df2)}")
    print(f"  - 唯一 SMILES 数: {len(set2)}")
    print(f"-"*30)
    print(f"重叠结果:")
    print(f"  - 共有 {overlap_count} 个 SMILES 同时存在于两个文件中")
    
    if len(set2) > 0:
        leakage_rate = (overlap_count / len(set2)) * 100
        print(f"  - 泄露率 (相对于文件 B): {leakage_rate:.2f}%")
    
    if overlap_count > 0:
        print(f"\n前 5 个重复的 SMILES 示例:")
        for s in list(intersection)[:5]:
            print(f"  {s}")
    print("="*50)

    return intersection

# 使用方法：替换为你实际的文件路径
train_csv = "feature-engineering_test_Morgan(1024)_Avalon(512)_ChemBERTa(384).csv"
test_csv = "feature-engineering_train_Morgan(1024)_Avalon(512)_ChemBERTa(384).csv"

overlap_smiles = check_smiles_overlap(train_csv, test_csv)