# 环境检查
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path.cwd().parent
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

print("✓ 所有库已成功导入")
print(f"✓ 项目根目录: {project_root}")



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


# ============== 参数配置区 ==============

CONFIG = {
    # 输入输出路径
    'processed_dir': project_root / 'data' / 'processed',
    'features_dir': project_root / 'outputs' / 'features',
    'cv_results_dir': project_root / 'outputs' / 'model_results' / 'phase3_binary' / 'cv_results',
    'feature_importance_dir': project_root / 'outputs' / 'model_results' / 'phase3_binary' / 'feature_importance',
    'transfer_results_dir': project_root / 'outputs' / 'model_results' / 'phase3_binary' / 'transfer_results',
    'figures_dir': project_root / 'outputs' / 'figures' / 'phase3',
    
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
print(f"  模型: {CONFIG['models_to_train']}")
print(f"  交叉验证折数: {CONFIG['n_folds']}")
print(f"  GPU加速: {CONFIG['use_gpu']}")
print(f"  XGBoost参数: max_depth={CONFIG['xgb_max_depth']}, lr={CONFIG['xgb_learning_rate']}")



def load_and_binarize_dataset(npz_path: Path, csv_path: Path, target: str):
    """
    加载数据并将标签二值化
    
    Args:
        npz_path: NPZ特征文件
        csv_path: 处理后的CSV文件（包含分钟标签）
        target: 'SIF' or 'SGF'
    
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
    
    # ID匹配
    id_to_idx = {str(id_): idx for idx, id_ in enumerate(ids_npz)}
    valid_indices = []
    valid_labels = []
    
    label_col = f"{target}_minutes"
    for _, row in df.iterrows():
        row_id = str(row['id'])
        if row_id in id_to_idx:
            label = row[label_col]
            if label != -1 and not pd.isna(label):
                valid_indices.append(id_to_idx[row_id])
                valid_labels.append(label)
    
    # 筛选有效样本
    X_valid = X[valid_indices]
    y_minutes = np.array(valid_labels)
    
    # 二值化：基于中位数
    median = np.median(y_minutes)
    y_binary = (y_minutes >= median).astype(int)  # 1=稳定, 0=不稳定
    
    print(f"  样本数: {len(X_valid)}")
    print(f"  中位数阈值: {median:.1f} 分钟")
    print(f"  稳定/不稳定: {np.sum(y_binary==1)}/{np.sum(y_binary==0)}")
    
    return X_valid, y_binary, median, feature_names

# 加载所有数据集
datasets_data = {}
npz_files = sorted(CONFIG['features_dir'].glob('*_processed.npz'))

print(f"加载并二值化 {len(npz_files)} 个数据集:\n")
for npz_file in npz_files:
    dataset_name = npz_file.stem.replace('_processed', '')
    csv_file = CONFIG['processed_dir'] / f"{dataset_name}_processed.csv"
    
    print(f"{dataset_name}:")
    
    # SIF
    X_sif, y_sif, median_sif, feat_names = load_and_binarize_dataset(npz_file, csv_file, 'SIF')
    print(f"  SIF完成")
    
    # SGF
    X_sgf, y_sgf, median_sgf, _ = load_and_binarize_dataset(npz_file, csv_file, 'SGF')
    print(f"  SGF完成\n")
    
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
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='binary', zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, average='binary', zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, average='binary', zero_division=0))
        
        # AUC（需要至少两个类别）
        if len(np.unique(y_test)) > 1:
            metrics['auc'].append(roc_auc_score(y_test, y_proba))
        else:
            metrics['auc'].append(np.nan)
    
    # 汇总结果
    results = {
        'dataset': dataset_name,
        'target': target,
        'model': model_name,
        'n_folds': n_folds,
        'metrics': metrics,
        'mean_metrics': {k: np.nanmean(v) for k, v in metrics.items()},
        'std_metrics': {k: np.nanstd(v) for k, v in metrics.items()},
    }
    
    return results

# 执行：批量交叉验证
print("开始交叉验证训练...\n")
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
            results = cross_validate_model(X, y, model_name, dataset_name, target)
            cv_results_all.append(results)
            
            # 保存结果
            result_path = CONFIG['cv_results_dir'] / f"{dataset_name}_{target}_{model_name}_cv.json"
            with open(result_path, 'w') as f:
                json.dump(convert_numpy_types(results), f, indent=2)
            
            print(f"F1={results['mean_metrics']['f1']:.4f} ✓")

print(f"\n✓ 交叉验证完成！共 {len(cv_results_all)} 个实验")
print(f"  结果已保存到: {CONFIG['cv_results_dir'].relative_to(project_root)}")


# 提取RF和XGBoost的特征重要性（只对一个数据集示例）
print("提取特征重要性...\n")

importance_count = 0
for dataset_name, data in datasets_data.items():
    for target in ['SIF', 'SGF']:
        X = data[f'X_{target.lower()}']
        y = data[f'y_{target.lower()}']
        feature_names = data['feature_names']
        
        if len(y) == 0:
            continue
        
        # RF特征重要性
        rf = get_model('rf', False)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        imp_path = CONFIG['feature_importance_dir'] / f"{dataset_name}_{target}_rf_importance.csv"
        importance_df.to_csv(imp_path, index=False)
        importance_count += 1
        
        # XGBoost特征重要性
        xgb = get_model('xgb', CONFIG['use_gpu'])
        xgb.fit(X, y)
        
        importance_df_xgb = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        imp_path_xgb = CONFIG['feature_importance_dir'] / f"{dataset_name}_{target}_xgb_importance.csv"
        importance_df_xgb.to_csv(imp_path_xgb, index=False)
        importance_count += 1

print(f"✓ 特征重要性提取完成！共 {importance_count} 个文件")
print(f"  结果已保存到: {CONFIG['feature_importance_dir'].relative_to(project_root)}")


def transfer_learning_test(X_train, y_train, X_test, y_test, model_name: str, use_gpu: bool = False):
    """
    迁移学习测试
    """
    model = get_model(model_name, use_gpu)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return metrics

# 执行：双向迁移学习
print("开始迁移学习测试...\n")
transfer_results_all = []
dataset_names = list(datasets_data.keys())

for i, train_dataset in enumerate(dataset_names):
    for j, test_dataset in enumerate(dataset_names):
        if i == j:  # 跳过自己
            continue
        
        print(f"{train_dataset} → {test_dataset}:")
        
        for target in ['SIF', 'SGF']:
            X_train = datasets_data[train_dataset][f'X_{target.lower()}']
            y_train = datasets_data[train_dataset][f'y_{target.lower()}']
            X_test = datasets_data[test_dataset][f'X_{target.lower()}']
            y_test = datasets_data[test_dataset][f'y_{target.lower()}']
            
            if len(y_train) == 0 or len(y_test) == 0:
                print(f"  {target}: 样本不足，跳过")
                continue
            
            print(f"  {target}:", end=" ")
            for model_name in CONFIG['models_to_train']:
                metrics = transfer_learning_test(X_train, y_train, X_test, y_test, model_name, CONFIG['use_gpu'])
                
                result = {
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset,
                    'target': target,
                    'model': model_name,
                    'metrics': metrics,
                }
                transfer_results_all.append(result)
                
                # 保存结果
                result_path = CONFIG['transfer_results_dir'] / f"{train_dataset}_to_{test_dataset}_{target}_{model_name}.json"
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"{model_name.upper()}(F1={metrics['f1']:.3f})", end=" ")
            print()

print(f"\n✓ 迁移学习完成！共 {len(transfer_results_all)} 个实验")
print(f"  结果已保存到: {CONFIG['transfer_results_dir'].relative_to(project_root)}")



print("="*70)
print("Phase 3: 模型验证 - 执行完毕")
print("="*70)

print("\n📁 生成的文件:")
print(f"\n  1. CV结果 ({len(list(CONFIG['cv_results_dir'].glob('*.json')))} 个JSON):")
print(f"     {CONFIG['cv_results_dir'].relative_to(project_root)}")

print(f"\n  2. 特征重要性 ({len(list(CONFIG['feature_importance_dir'].glob('*.csv')))} 个CSV):")
print(f"     {CONFIG['feature_importance_dir'].relative_to(project_root)}")

print(f"\n  3. 迁移学习结果 ({len(list(CONFIG['transfer_results_dir'].glob('*.json')))} 个JSON):")
print(f"     {CONFIG['transfer_results_dir'].relative_to(project_root)}")

print(f"\n  4. 可视化图表 ({len(list(CONFIG['figures_dir'].glob('*.png')))} 个PNG):")
for f in sorted(CONFIG['figures_dir'].glob('*.png')):
    print(f"     - {f.name}")

# 找出最佳模型
best_cv = max(cv_results_all, key=lambda x: x['mean_metrics']['f1'])
print("\n🏆 最佳CV模型:")
print(f"  数据集: {best_cv['dataset']}")
print(f"  目标: {best_cv['target']}")
print(f"  模型: {best_cv['model'].upper()}")
print(f"  F1 Score: {best_cv['mean_metrics']['f1']:.4f} ± {best_cv['std_metrics']['f1']:.4f}")
print(f"  Accuracy: {best_cv['mean_metrics']['accuracy']:.4f} ± {best_cv['std_metrics']['accuracy']:.4f}")

best_transfer = max(transfer_results_all, key=lambda x: x['metrics']['f1'])
print("\n🌐 最佳迁移学习模型:")
print(f"  训练集: {best_transfer['train_dataset']}")
print(f"  测试集: {best_transfer['test_dataset']}")
print(f"  目标: {best_transfer['target']}")
print(f"  模型: {best_transfer['model'].upper()}")
print(f"  F1 Score: {best_transfer['metrics']['f1']:.4f}")

print("\n📊 总体统计:")
print(f"  交叉验证实验: {len(cv_results_all)}")
print(f"  迁移学习实验: {len(transfer_results_all)}")
print(f"  特征重要性分析: {importance_count}")

print("\n✅ Phase 3 完成！所有结果已保存")
print("="*70)