# 根据feature_importance_summary.json中的内容
# 在上述文件中，根据permutation importance的方式获取特征的数值
# 其中，mean值>0为有效特征，=0为不拖后腿的，<0为拖后腿的
# 这里提供了两种形式，一种是仅大于0的视为有效（这种只会获取其中一部分特征）
#                   另一种是大于等于0的均视为有效

# 配置设置：在此处直接进行修改，将在下方执行对应的脚本逻辑================================================
CONFIG={
    'tragedy':'without_0',   #两种不同的策略模式， with_0 或者 without_0, 代表保留的时候是否需要保留0
    'file_path':"feature_importance_summary.json", #文件的读入路径
    'output_path':'importance_mask.json',  #文件的输出路径
}


# 运行代码============================================================================================

import json
import re

def process_feature_importance_json(file_path):
    """
    处理特征重要性JSON文件，提取指定格式的键名，整合特征重要性数据，并生成掩码数组
    
    Args:
        file_path (str): JSON文件路径
    """
    
    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 用于存储分组后的数据
    grouped_data = {}
    
    # 第一步：按照倒数第二个_后面的内容进行分组
    for key, value in data.items():
        # 使用正则表达式提取倒数第二个_后面的内容
        parts = key.split('_')
        if len(parts) >= 2:
            # 提取最后两部分并连接
            group_key = f"{parts[-2]}_{parts[-1]}"
            
            # 初始化该组的列表（如果尚未存在）
            if group_key not in grouped_data:
                grouped_data[group_key] = []
            
            # 提取所有feature_{}的importance_mean值
            for feature_key, feature_value in value.items():
                if feature_key.startswith('feature_') and 'importance_mean' in feature_value:
                    grouped_data[group_key].append(feature_value['importance_mean'])
    
    # 第二步：输出检查整合后的数组
    print("=== 整合后的特征重要性数组 ===")
    for group_key, importance_array in grouped_data.items():
        print(f"{group_key}: {importance_array}")
    
    # 第三步：生成掩码数组,该部分根据tragedy的情况，分为两种型态
    print("\n=== 生成的掩码数组 ===")
    mask_arrays = {}
    if CONFIG['tragedy']=='with_0':
        for group_key, importance_array in grouped_data.items():
            mask_array = [1 if x >= 0 else 0 for x in importance_array]
            mask_arrays[group_key] = mask_array
            print(f"{group_key}: {mask_array}")
    
        # 第四步：统计信息
        print("\n=== 统计信息 ===")
        for group_key, importance_array in grouped_data.items():
            total_features = len(importance_array)
            positive_features = sum(1 for x in importance_array if x >= 0)
            negative_features = total_features - positive_features
            mask_ones = sum(mask_arrays[group_key])
            mask_zeros = total_features - mask_ones
            
            print(f"{group_key}:")
            print(f"  总特征数: {total_features}")
            print(f"  重要性≥0的特征: {positive_features} ({positive_features/total_features*100:.1f}%)")
            print(f"  重要性<0的特征: {negative_features} ({negative_features/total_features*100:.1f}%)")
            print(f"  掩码中1的数量: {mask_ones} ({mask_ones/total_features*100:.1f}%)")
            print(f"  掩码中0的数量: {mask_zeros} ({mask_zeros/total_features*100:.1f}%)")
    elif CONFIG['tragedy']=='without_0':
        for group_key, importance_array in grouped_data.items():
            mask_array = [1 if x > 0 else 0 for x in importance_array]
            mask_arrays[group_key] = mask_array
            print(f"{group_key}: {mask_array}")
    
        # 第四步：统计信息
        print("\n=== 统计信息 ===")
        for group_key, importance_array in grouped_data.items():
            total_features = len(importance_array)
            positive_features = sum(1 for x in importance_array if x > 0)
            negative_features = total_features - positive_features
            mask_ones = sum(mask_arrays[group_key])
            mask_zeros = total_features - mask_ones
            
            print(f"{group_key}:")
            print(f"  总特征数: {total_features}")
            print(f"  重要性>0的特征: {positive_features} ({positive_features/total_features*100:.1f}%)")
            print(f"  重要性≤0的特征: {negative_features} ({negative_features/total_features*100:.1f}%)")
            print(f"  掩码中1的数量: {mask_ones} ({mask_ones/total_features*100:.1f}%)")
            print(f"  掩码中0的数量: {mask_zeros} ({mask_zeros/total_features*100:.1f}%)")
    else:
        grouped_data=[]
        mask_array=[]
        print("策略输入失误, 不进行特征筛选的话可以在其他脚本中设置")
    

    
    return grouped_data, mask_arrays

def save_results(grouped_data, mask_arrays, output_file):
    """
    保存处理结果到JSON文件
    
    Args:
        grouped_data (dict): 分组后的特征重要性数据
        mask_arrays (dict): 掩码数组数据
        output_file (str): 输出文件路径
    """
    results = {
        'grouped_importance': grouped_data,
        'mask_arrays': mask_arrays
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_file = CONFIG['file_path']  # 你的输入文件
    output_file = CONFIG['output_path']   # 输出文件
    
    try:
        # 处理数据
        grouped_data, mask_arrays = process_feature_importance_json(input_file)
        
        # 保存结果
        save_results(grouped_data, mask_arrays, output_file)
        
        print("\n=== 处理完成 ===")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误: {input_file} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")