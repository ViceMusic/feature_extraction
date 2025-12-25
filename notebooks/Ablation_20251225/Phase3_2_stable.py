# 这段代码原本的来源是同文件夹下的Phase3_1_stable.ipynb笔记本，原本只进行一次预测就输出结果
# 目前已经修改为变化100次随机种子，并且绘制迁移学习热力图


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

# 交叉验证结果已经验证完了，现在开始验证别的，比如迁移学习
def transfer_learning_test(X_train, y_train, X_test, y_test, model_name: str, use_gpu: bool = False):
    """
    迁移学习测试
    """
    model = get_model(model_name, use_gpu)

            # 排除特殊情况
    try:
        # xgb.fit(X, y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"由于标签问题无法成功进行迁移学习测试，跳过该任务{e}")
        return None
        
    
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return metrics


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

import json
import numpy as np

import json
import numpy as np
from pathlib import Path

print("开始迁移学习测试...\n")
transfer_results_all = []
dataset_names = list(datasets_data.keys())
num_runs = 100  # 迁移学习重复次数

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
                metrics_list = []

                # 执行 num_runs 次迁移学习
                for run_id in range(num_runs):
                    CONFIG['random_state'] = CONFIG.get('random_state', 42) + run_id  # 每次运行不同随机种子
                    metrics = transfer_learning_test(X_train, y_train, X_test, y_test, model_name, CONFIG['use_gpu'])
                    if metrics is not None:
                        metrics_list.append(metrics)
                
                # 如果没有有效结果，跳过
                if len(metrics_list) == 0:
                    avg_metrics = None
                    print(f"{model_name.upper()}(跳过) ", end=" ")
                else:
                    # 对每个 metric 求平均（支持标量或数组）
                    avg_metrics = {}
                    for key in metrics_list[0].keys():
                        values = []
                        for m in metrics_list:
                            v = m[key]
                            if v is None:
                                continue
                            if isinstance(v, (np.ndarray, list)):
                                v = np.array(v).flatten()
                                v = v[~np.isnan(v)]
                                if len(v) > 0:
                                    values.append(np.mean(v))
                            else:  # 标量
                                if not np.isnan(v):
                                    values.append(v)
                        avg_metrics[key] = float(np.mean(values)) if len(values) > 0 else None

                    print(f"{model_name.upper()}(F1={avg_metrics['f1']:.3f})", end=" ")

                # 构造结果字典
                result = {
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset,
                    'target': target,
                    'model': model_name,
                    'metrics': avg_metrics,
                }
                transfer_results_all.append(result)
                
                # 保存 JSON
                result_path = CONFIG['transfer_results_dir'] / f"{train_dataset}_to_{test_dataset}_{target}_{model_name}.json"
                Path(result_path).parent.mkdir(parents=True, exist_ok=True)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
            print()

print(f"\n✓ 迁移学习完成！共 {len(transfer_results_all)} 个实验")
print(f"  结果已保存到: {CONFIG['transfer_results_dir'].relative_to(project_root)}")


# 生成迁移学习热力图
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# 迁移学习热力图（6个子图横向，只有最右侧显示 colorbar，第一个子图显示坐标轴标签）
def plot_transfer_learning_heatmaps(
    transfer_results_all,
    targets=('SIF', 'SGF'),
    models=('lr', 'rf', 'xgb'),
    highlight_pairs=[('US9624268', 'WO2017011820A2'), ('WO2017011820A2', 'US9624268')],
    save_dir="transfer_heatmaps",
    fig_name="transfer_learning.png",
    task_name="Task",
    dpi=300,
    display_plots=False
):
    """
    生成迁移学习 F1 热力图，6 个子图横向排列
    子标题格式：{task}-{模型}
    坐标轴标签只在第一个子图显示
    仅最右侧子图显示 colorbar
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    n_plots = len(targets) * len(models)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 5),  # 微微调大热力图
        constrained_layout=True
    )
    
    if n_plots == 1:
        axes = [axes]
    
    plot_id = 0
    for target in targets:
        for model in models:
            # 构建 F1 Score 矩阵
            transfer_df = pd.DataFrame([
                {
                    'train': r['train_dataset'],
                    'test': r['test_dataset'],
                    'f1': r['metrics']['f1'] if r['metrics'] is not None else np.nan
                }
                for r in transfer_results_all
                if r['target'] == target and r['model'] == model
            ])

            if len(transfer_df) == 0:
                plot_id += 1
                continue

            heatmap_data = transfer_df.pivot(
                index='train',
                columns='test',
                values='f1'
            )

            nan_mask = heatmap_data.isna()    # NaN → 空白
            zero_mask = heatmap_data == 0     # 0 → 灰色

            ax = axes[plot_id]

            # 判断是否显示 colorbar：仅最后一个子图显示
            show_cbar = (plot_id == n_plots - 1)
            # 判断是否显示坐标轴标签：仅第一个子图显示
            show_labels = (plot_id == 0)

            # 第一层：正常 F1
            sns.heatmap(
                heatmap_data,
                mask=nan_mask | zero_mask,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                square=True,
                ax=ax,
                cbar=show_cbar,
                cbar_kws={'label': 'F1 Score', 'shrink':0.7} if show_cbar else None,
                xticklabels=True,
                yticklabels=True
            )

            # 第二层：F1 == 0 → 特殊灰色
            sns.heatmap(
                heatmap_data,
                mask=~zero_mask,
                annot=True,
                fmt='.3f',
                cmap=sns.color_palette(['#B0B0B0']),
                square=True,
                cbar=False,
                ax=ax
            )

            # 加粗框：指定数据集双向迁移
            for train_ds, test_ds in highlight_pairs:
                if train_ds in heatmap_data.index and test_ds in heatmap_data.columns:
                    y = heatmap_data.index.get_loc(train_ds)
                    x = heatmap_data.columns.get_loc(test_ds)
                    rect = plt.Rectangle(
                        (x, y), 1, 1,
                        fill=False,
                        edgecolor='black',
                        linewidth=2.0
                    )
                    ax.add_patch(rect)

            # 子标题、小字体
            ax.set_title(f"{task_name}: {target}-{model.upper()}", fontsize=9)

            # 坐标轴标签仅第一个子图显示
            if show_labels:
                ax.set_xlabel('Test Set', fontsize=20)
                ax.set_ylabel('Train Set', fontsize=20)
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')

            # 微调刻度字体
            ax.tick_params(axis='x', labelsize=12, rotation=45)
            ax.tick_params(axis='y', labelsize=12)

            plot_id += 1

    # 保存文件
    save_path = Path(save_dir) / fig_name
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ 已保存: {save_path.name}")

    if display_plots:
        plt.show()
    else:
        plt.close()


# 调用示例
print("\n生成迁移学习热力图...")

epoch="All"

plot_transfer_learning_heatmaps(
    transfer_results_all=transfer_results_all,
    targets=['SIF', 'SGF'],
    models=['lr', 'rf', 'xgb'],
    highlight_pairs=[('US9624268', 'WO2017011820A2'), ('WO2017011820A2', 'US9624268')],
    save_dir="trans_figures",
    fig_name=f"transfer_learning_heatmaps_{epoch}.png",
    task_name=f"{epoch}",
    dpi=300,
    display_plots=True
)

print("✓ 迁移学习热力图生成完成！")
