# 这段代码原本的来源是同文件夹下的Phase3_1_stable.ipynb笔记本，原本只进行一次预测就输出结果
# 目前已经修改为会进行一百次的交叉验证，然后将结果移动到json文件中（统计每一次执行的metric_stds和means）
# 后续可以配合clean_cv_json.py和aggregate_cv_json.py进行清洗和均值计算
# 最终生成一个均值json文件，然后配合generate_cv_figure脚本绘制热力


# 检查模型导入情况：
# 环境检查
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path.cwd().parent   # 这里得到的是python进程的工作目录！而不是脚本所在位置，除非在终端重就已经进入脚本所在目录
print(project_root)
sys.path.insert(0, str(project_root / "src"))

# 核心库导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

# 机器学习
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
# 检查GPU可用性
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU不可用，将使用CPU训练")
except ImportError:
    gpu_available = False
    print("⚠ PyTorch未安装，将使用CPU训练")
# 设置显示选项
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# 导入配置内容：测试如下
# ============== 参数配置区 ==============

CONFIG = {
    # 输入路径
    'processed_dir': project_root / 'data' / 'processed',
    'features_dir': project_root / 'outputs' / 'features',
    # 输出路径
    'cv_results_dir': project_root / 'notebooks'/'Phase3_1_temp'/  'model_results' / 'phase3_binary' / 'cv_results',
    'feature_importance_dir': project_root / 'notebooks'/'Phase3_1_temp'/  'model_results' / 'phase3_binary' / 'feature_importance',
    'transfer_results_dir': project_root / 'notebooks'/'Phase3_1_temp'/  'model_results' / 'phase3_binary' / 'transfer_results',
    'figures_dir': project_root / 'notebooks'/'Phase3_1_temp'/ 'figures' / 'phase3',

    # 模型选择（可选: 'lr', 'rf', 'xgb'）
    'models_to_train': ['lr', 'rf', 'xgb'],
    
    # 交叉验证参数
    'n_folds': 5,
    'random_state': 42,
    
    # XGBoost参数
    'use_gpu': gpu_available,
    'xgb_max_depth': 6,
    'xgb_learning_rate': 0.1,
    'xgb_n_estimators': 100,
    
    # Random Forest参数
    'rf_n_estimators': 100,
    'rf_n_jobs': -1,
    
    # Logistic Regression参数
    'lr_max_iter': 1000,
    
    # 可视化参数
    'dpi': 300,
    'format': 'png',
    'display_plots': True,
    'max_display_plots': 8,
}

# 创建输出目录
for key in ['cv_results_dir', 'feature_importance_dir', 'transfer_results_dir', 'figures_dir']:
    CONFIG[key].mkdir(parents=True, exist_ok=True)

print("配置参数:")
print(f"  随机种子: {CONFIG['random_state']}")
print(f"  模型: {CONFIG['models_to_train']}")
print(f"  交叉验证折数: {CONFIG['n_folds']}")
print(f"  GPU加速: {CONFIG['use_gpu']}")
print(f"  XGBoost参数: max_depth={CONFIG['xgb_max_depth']}, lr={CONFIG['xgb_learning_rate']}")

# 加载并且获取数据集
def load_and_binarize_dataset(npz_path: Path, csv_path: Path, target: str, is_monomer: bool = None):
    """
    加载数据并将标签二值化，同时可选择只保留monomer或非monomer样本
    (2025-12-15新增功能)
    
    Args:
        npz_path: NPZ特征文件
        csv_path: 处理后的CSV文件（包含分钟标签）
        target: 'SIF' or 'SGF'
        is_monomer: 如果为True，只保留monomer；False，只保留非monomer；None不筛选
    
    Returns:
        X, y_binary, median_threshold, feature_names
    """
    # 加载NPZ特征
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    feature_names = data['feature_names']
    ids_npz = data['ids']
    
    # 加载CSV获取分钟标签
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str)
    
    # ID匹配：这里是建立一个映射数值
    id_to_idx = {str(id_): idx for idx, id_ in enumerate(ids_npz)}
    valid_indices = [] # 确定当前任务用哪一行
    valid_labels = []
    
    label_col = f"{target}_minutes" # 根据任务目标获取分钟数目
    for _, row in df.iterrows():
        row_id = str(row['id'])
        if row_id in id_to_idx:
            #===================================过滤无效标签=========================================
            label= row[label_col]
            if label == -1 or pd.isna(label):# 判断标签有效（对应的任务就只能做对应的二值化）
                continue
            if is_monomer is not None and row['is_monomer'] != is_monomer:  # 判断是否满足monomer条件
                continue
            if row[f"{target}_minutes"]>700: #特殊情况处理，如果是SIF_minuters>700
                continue   
            #=======================================================================================
            # 符合条件则加入
            valid_indices.append(id_to_idx[row_id]) # 获取对应数据的索引/序号
            valid_labels.append(label)              # 获取label
    
    # 筛选有效样本
    X_valid = X[valid_indices]   # 根据序号直接获取到值，这里是numpy的基本用法之一，比如传入【3，7】那么得到的就是第三行和第七行的数值
    y_minutes = np.array(valid_labels)  # 获取y值

    
    
    # 设定阈值========================
    median =None
    if target=='SIF':
        median = 270
    elif target=='SGF':
        median =250#（经过纠正250更好点）
    
    # 根据阈值判断是否稳定捏
    y_binary = (y_minutes >= median).astype(int)  # 1=稳定, 0=不稳定
    
    print(f"  样本数: {len(X_valid)}")
    print(f"  中位数阈值（更新后阈值根据75%划分得出结果）: {median:.1f} 分钟")
    print(f"  稳定/不稳定: {np.sum(y_binary==1)}/{np.sum(y_binary==0)}")
    
    return X_valid, y_binary, median, feature_names


# 加载所有数据集
datasets_data = {}
npz_files = sorted(CONFIG['features_dir'].glob('*_processed.npz'))
print(f"发现 {CONFIG['features_dir']} 个处理后的数据集文件。")
# 选择加载数据的时候就指定monomer或非monomer样本
is_monomer = True # 仅加载monomer样本，设置为False则加载非monomer样本，None则不筛选

print(f"加载并二值化 {len(npz_files)} 个数据集:\n")
for npz_file in npz_files:
    dataset_name = npz_file.stem.replace('_processed', '')
    csv_file = CONFIG['processed_dir'] / f"{dataset_name}_processed.csv"
    
    print(f"{dataset_name}:")

    # 此处新增数据集加载功能（第四个参数）
    
    # SIF
    X_sif, y_sif, median_sif, feat_names = load_and_binarize_dataset(npz_file, csv_file, 'SIF',is_monomer)
    print(f"  SIF数据选择完成，只获得单体数据")
    
    # SGF
    X_sgf, y_sgf, median_sgf, _ = load_and_binarize_dataset(npz_file, csv_file, 'SGF',is_monomer)
    print(f"  SGF数据选择完成，只获取单体数据\n")
    
    datasets_data[dataset_name] = {
        'X_sif': X_sif,
        'y_sif': y_sif,
        'median_sif': median_sif,
        'X_sgf': X_sgf,
        'y_sgf': y_sgf,
        'median_sgf': median_sgf,
        'feature_names': feat_names,
    }

print(f"✓ 数据加载完成！共 {len(datasets_data)} 个数据集")

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
# -----------------上述内容已经准备完毕，数据集读取成功-----------------
# 遇到的一个问题是读取数据时候的相对路径问题，整个python和notebook理论上处理逻辑是一样的，但是本身原因出在“进程执行所在目录”，终端和notebook是完全不同的逻辑
#--------------------------------------------------------------------------

# 进行交叉验证工作：其中验证结果存储在其内容中
def get_model(model_name: str, use_gpu: bool = False):
    """
    创建模型实例
    """
    if model_name == 'lr':
        return LogisticRegression(
            max_iter=CONFIG['lr_max_iter'],
            class_weight='balanced',
            random_state=CONFIG['random_state']
        )
    elif model_name == 'rf':
        return RandomForestClassifier(
            n_estimators=CONFIG['rf_n_estimators'],
            class_weight='balanced',
            n_jobs=CONFIG['rf_n_jobs'],
            random_state=CONFIG['random_state']
        )
    elif model_name == 'xgb':
        params = {
            'max_depth': CONFIG['xgb_max_depth'],
            'learning_rate': CONFIG['xgb_learning_rate'],
            'n_estimators': CONFIG['xgb_n_estimators'],
            'random_state': CONFIG['random_state'],
            'tree_method': 'hist',
        }
        if use_gpu:
            params['device'] = 'cuda:0'
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def cross_validate_model(X, y, model_name: str, dataset_name: str, target: str):
    """
    执行k折交叉验证
    
    Returns:
        dict: CV结果
    """
    # 自动调整fold数（小数据集）
    min_class_count = np.bincount(y).min()
    n_folds = min(CONFIG['n_folds'], min_class_count)
    if n_folds < CONFIG['n_folds']:
        print(f"    ⚠ 样本数较少，调整fold数为 {n_folds}")
    
    if n_folds < 2:
        print("    ⚠ 样本过少，无法进行分层交叉验证，跳过该任务")
        return None

    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CONFIG['random_state'])
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        model = get_model(model_name, CONFIG['use_gpu'])
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        # （修改后的probo）
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            y_proba = proba[:, 1]
        else:
            y_proba = None  # 只有一个类别时，AUC 没有定义
        
        # 计算指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='binary', zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, average='binary', zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, average='binary', zero_division=0))
        
        # AUC（需要至少两个类别，新增一种情况与下面配合，就是只有一种类别）
        if y_proba is not None and len(np.unique(y_test)) > 1:
            metrics['auc'].append(roc_auc_score(y_test, y_proba))
        else:
            metrics['auc'].append(np.nan)
    
    # 汇总结果
    '''
        这是一个大致的输出结果------------------metrics部分是多次计算的结果，也是个字典
        "dataset": "WO2017011820A2",
        "target": "SIF",
        "model": "lr",
        其实可以只要
        'dataset': dataset_name,
        'target': target,
        'model': model_name,

        'n_folds': n_folds,
        'mean_metrics': {k: np.nanmean(v) for k, v in metrics.items()},
        'std_metrics': {k: np.nanstd(v) for k, v in metrics.items()},
    '''
    results = {
        'dataset': dataset_name,
        'target': target,
        'model': model_name,
        'n_folds': n_folds,
        'metrics': metrics, # 这个其实就是多次的计算结果
        'mean_metrics': {k: np.nanmean(v) for k, v in metrics.items()},
        'std_metrics': {k: np.nanstd(v) for k, v in metrics.items()},
    }
    
    return results

# 执行：批量交叉验证
print("开始交叉验证训练...\n")
cv_means_all={
    'SIF':{
        'lr': {
            'sif_sgf_second':{
                'means': [],
                'stds': []
            },
            'US9624268':{
                'means': [],
                'stds': []
            },
            'US9809623B2':{
                'means': [],
                'stds': []
            },
            'US20140294902A1':{
                'means': [],
                'stds': []
            },
            'WO2017011820A2':{
                'means': [],
                'stds': []
            },
        },
        'rf': {
                        'sif_sgf_second':{
                'means': [],
                'stds': []
            },
            'US9624268':{
                'means': [],
                'stds': []
            },
            'US9809623B2':{
                'means': [],
                'stds': []
            },
            'US20140294902A1':{
                'means': [],
                'stds': []
            },
            'WO2017011820A2':{
                'means': [],
                'stds': []
            },
        },
        'xgb': {
                        'sif_sgf_second':{
                'means': [],
                'stds': []
            },
            'US9624268':{
                'means': [],
                'stds': []
            },
            'US9809623B2':{
                'means': [],
                'stds': []
            },
            'US20140294902A1':{
                'means': [],
                'stds': []
            },
            'WO2017011820A2':{
                'means': [],
                'stds': []
            },
        }
    },
    'SGF':{
        'lr': {
           'sif_sgf_second':{
                'means': [],
                'stds': []
            },
            'US9624268':{
                'means': [],
                'stds': []
            },
            'US9809623B2':{
                'means': [],
                'stds': []
            },
            'US20140294902A1':{
                'means': [],
                'stds': []
            },
            'WO2017011820A2':{
                'means': [],
                'stds': []
            },
        },
        'rf': {
                        'sif_sgf_second':{
                'means': [],
                'stds': []
            },
            'US9624268':{
                'means': [],
                'stds': []
            },
            'US9809623B2':{
                'means': [],
                'stds': []
            },
            'US20140294902A1':{
                'means': [],
                'stds': []
            },
            'WO2017011820A2':{
                'means': [],
                'stds': []
            },
        },
        'xgb': {
                        'sif_sgf_second':{
                'means': [],
                'stds': []
            },
            'US9624268':{
                'means': [],
                'stds': []
            },
            'US9809623B2':{
                'means': [],
                'stds': []
            },
            'US20140294902A1':{
                'means': [],
                'stds': []
            },
            'WO2017011820A2':{
                'means': [],
                'stds': []
            },
        }
    }
}
# 这里的格式为 任务-模型-数据集-（means_list和stds_list）
cv_results_all = []

for dataset_name, data in tqdm(datasets_data.items(), desc="数据集"):
    print(f"\n{dataset_name}:")
    
    for target in ['SIF', 'SGF']:
        X = data[f'X_{target.lower()}']
        y = data[f'y_{target.lower()}']
        
        if len(y) == 0:
            print(f"  {target}: 无有效样本，跳过")
            continue
        
        print(f"  {target}:")
        for model_name in CONFIG['models_to_train']:
            print(f"    {model_name.upper()}...", end=" ")

            # 这里整个的训练过程最好是重复个一百次，并且每次生成一个随机种子
            # 提取其中多次计算的means和stds，存储为数组，然后统计
            # 生成一个json吧，在不影响原本的情况下
            # 任务-模型-数据集，每个情况都整理一个CV曲线出来

            #=====增加错误排除逻辑===========================
            for i in range(100):
                try:
                    ## 这里应该是重复100次，
                    CONFIG['random_state']+=1 # 每次增加随机种子
                    results = cross_validate_model(X, y, model_name, dataset_name, target)
                    print(results)
                    if results is not None:
                    # 填入信息
                        mean=results['mean_metrics']
                        std=results['std_metrics']
                        cv_means_all[target][model_name][dataset_name]['means'].append(mean)
                        cv_means_all[target][model_name][dataset_name]['stds'].append(std)
                    else:
                        cv_means_all[target][model_name][dataset_name]['means'].append(None)
                        cv_means_all[target][model_name][dataset_name]['stds'].append(None)
                except ValueError as e:
                    print(f"只有一种类别，跳过该任务")
                    cv_means_all[target][model_name][dataset_name]['means'].append(None)
                    cv_means_all[target][model_name][dataset_name]['stds'].append(None)
                    continue
                # 增加跳出逻辑====================================
                if results is None:
                    print("跳过 ✓")
                    cv_means_all[target][model_name][dataset_name]['means'].append(None)
                    cv_means_all[target][model_name][dataset_name]['stds'].append(None)
                    continue
                #===============================================

import json

with open("NoAll.json", "w", encoding="utf-8") as f:
    json.dump(cv_means_all, f, ensure_ascii=False, indent=2)

print(f"\n✓ 交叉验证完成！共 {len(cv_results_all)} 个实验")
print(f"  结果已保存到: {CONFIG['cv_results_dir'].relative_to(project_root)}")