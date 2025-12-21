# 全局文件的安装配置

## 关于UV和NOTEBOOK

pip一般是全局安装，会导致依赖地狱，这就是虚拟环境的方法

venv可以创建虚拟环境 python3 -m venv .vene(接下来就需要激活该环境)
本质上是修改sys.path里面的东西
卸载的时候也很抽象，间接依赖也会导致出问题

pyproject.toml是个官方方案，这里面记录了所有需要的库，UV也是通过这个文件来确定需要安装什么库，不过直接使用UV即可，UV会自动激活虚拟环境.venv

安装好UV其实就可以正常使用NOTEBOOK的内容了

## UV的使用和下载

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

然后再控制台输入UV，输出字样则表示安装成功

uv python list 可以检查当前python版本

    常用指令
    初始化项目	uv init
    从 toml 安装依赖	uv sync
    安装单个库	uv add xxx
    删除库	uv remove xxx
    临时运行脚本	uv run xxx.py
    执行命令	uv run （python） main.py

举例子
```
uv run main.py
uv run python --version # 该命令用于初始化虚拟环境并且下载所有库
```


## 关于这个项目里的notebook怎么使用

首先进入ipynb文件中可以单独执行不同的cell，执行结果会放在cell下方。

如果想要执行到某个cell为止，就run above all，会执行当前cell前面所有的cell（不包括该cell）

然后如果遇到kernel死掉的问题可以重启vscode解决


# 20251217总结日志


1. 参考之前github的代码，把phase1中 Morgan_fingerprint 去除后再跑后续的phase2/3
2. phase3训练模型前注意增加几项处理，只考虑monomer，去除monomer数据，并且把US9809623B2数据集中的异常值去除
3. 训练模型，给出定量的性能比较，去除Morgan_fingerprint前后有什么性能差别
4. 做几页PPT呈现必要的结果，务必让我很容易理解你做了什么，结果是什么，得到什么结论

## 0. 补充知识点

Morgan指纹是一种用向量展示模型的手段，会把分子的序列结构进行破坏，关闭Morgan的目的是强迫模型关注本身的物理属性而非hash pattern。

CONFIG['morgan_bits'] = 0 代码是“开关”，为每个分子生成n维度的特征

在phase1中，data_filtering_summary.png的横轴为数据集，纵轴为样本数（左边是样本数，右边是过滤以后的样本数目）

另一张图，是表达保留样本有多少，每个样本里面还剩下啥

```
ax.bar(x, dimer_pct, label='二聚体 %')
ax.bar(x, cyclic_pct, bottom=dimer_pct, label='环化 %')
ax.bar(x, disulfide_pct, bottom=dimer_pct + cyclic_pct, label='二硫键 %')
```

从上到下 **不是互斥的类别**，而是**结构属性占比**

数据处理后的结果：

- `data/processed/*.csv` - 添加分子特征后的CSV  
- `outputs/features/*.npz` - RDKit特征矩阵  
- `outputs/figures/phase1/*.png` - 质量验证图表  


一些数据的意义
-  is_monomer：是否为单体
- `is_dimer`: 是否为二聚体 (bool)
- `is_cyclic`: 是否含环状结构 (bool)
- `has_disulfide_bond`: 是否含二硫键 (bool)
- `SIF_minutes`: SIF半衰期（分钟）
- `SGF_minutes`: SGF半衰期（分钟）

fold是K折交叉验证的K的意思，默认参数在phase3中设置为5

## 1. 核心流程划分

Phase 1: 特征提取与预处理

    输入：分子的 SMILES 序列。
    输出：清洗后的 .csv、特征矩阵 .npz 以及质量验证图表 .png。
    关键配置：对比了开启/关闭 Morgan 指纹（1024 bits）的特征差异。

Phase 2: 统计验证

    验证单体/二聚体差异，检查数据集间的异质性，确保特征的生物合理性。
    进行统计和可视化操作

Phase 3: 模型训练与评估

    模型：Logistic Regression (LR), Random Forest (RF), XGBoost (XGB)。
    任务类型：SGF， SIF半衰期
    验证方案：5-Fold 交叉验证 (CV) + 跨数据集迁移学习测试。
    改进点：新增 is_monomer 筛选功能，支持剔除特定数据集（如 US9809623B2）的异常值。

## 2. 实验操作点，以及相关设置信息

### 2.1 根据is_monomer字段进行筛选

数据集中携带bool类型的is_monomer字段，代表该分子是否为单体。

但是部分数据集如果选择过滤掉monomer的话，会导致后续分类任务报错，已经在代码中完成了修改，这部分不会报错而是会返回None，并且在实际任务中，只要US964268和WO2017011820A2两个数据集能正常运行任务即可，这两个数据集被验证是存在迁移能力的。

### 2.2 阈值 / 异常值过滤

根据直方图分析，两种任务的阈值/异常值如下

SIF： 阈值为270，异常值为700
SGF： 阈值为250，异常值为700


### 2.3 实验目的

三种模型在两种不同的任务场景下（相当于一共六种情况）
检查去除Morgan分子指纹后模型性能的变化，迁移能力是否提升。

## 3 实验结论

本次工作的实验结论证实：

### 3.1 去除Morgan分子指纹后的情况

去除Morgan分子指纹后，模型的正确率略微下降但不明显，数据产生的稳定性有所提升

不过这也收到特定任务的影响，在SGF任务下使用随机森林模型时，这种情况最清晰

### 3.2 数据集情况

一共有五个数据集，其中US20140294902A1数据集是用来测试的，不纳入考虑
sif_sgf_second数据集主要用来跑流程，本身意义不大

最重要的是4268和WO2017两个数据集，这两个数据集能跑通所有任务，并且具备相互迁移的能力

US9809623B2数据集在SGF任务中也有一定的迁移潜力，但是该数据集的波动情况很大，可能是比较敏感。

### 3.3 实验留档

    D:\RA\feature_extraction\recordings\20251216数据分析整理.pdf


# 20251221日报

备注：
    前两天考托业去了，有点忙

**1.更新内容**
* 新建了get_npy_or_pkl.ipynb文件，主要内容为生成processed文件和npy文件（在这一阶段就对单体分子进行筛选）
* 新建了read_npy.ipynb文件，主要内容为读取npy文件，并且根据数据集-任务筛选，根据阈值二值化y，剔除异常值，最终返回nbarry格式数据（X，二值化y，原始y，特征名称）

**2.修改问题**
* 1.在上述更新内容中，修改了路径读取问题，由于不确定"运行环境"到底指的是什么，这里将其修改为如下格式， 直接获取”当前脚本所在位置“作为根目录，后续的新增脚本将全部跟随该思路。
    ```
    import numpy as np
    from pathlib import Path

    # 获取相对目录的两种方式，前者针对正常环境，后者针对notebook
    script_dir = Path(__file__).parent  # 脚本目录,但是这个在notebook里面不能用
    script_dir = Path().resolve()  # 当前文件所在目录

    # 类似这种用法，对于Path类路径，可以直接使用 / 进行拼接
    data_path = script_dir / "../outputs/npy_datas/sif_sgf_second_processed/X.npy"

    # 进行后续操作
    X = np.load(data_path)
    ```

* 2.关于分子指纹

    分子指纹的表现形式为布尔值集合，例如Morgan的分子指纹（1024bit），在提取后的特征中表现为

    ```
    [0,1,0,2,1,..........]# 1024个值，每个数值代表一个特征是否存在
    ```

    Avlvon也是类似的处理方式

**3.关于numpy可用的mask操作**

* 对于numpy数据来说，可以使用dtype为bool的数组，对ndarray格式的数据进行筛选（前提是mask数组和原始数组长度一致）
* 对于mask数组，可以使用初始化为1，长度为n的初始值，使用按位与操作，为每个需要被筛掉的位置变更为False
* X[mask]机制完成操作，如下方代码所示

    ```
        # ------------------------------------------------------------------
        # 4和5用了三个很有意思的特性， csv转npy时候的自动转化操作，numpy的筛选操作，多重叠加的按位与操作
        # ------------------------------------------------------------------

        # 4. 构造样本筛选 mask
        valid_mask = np.ones(n, dtype=bool)  # 生成的原址数值全都为1

        # 4.1 根据 is_monomer 过滤（如果指定），该方法已经被完全废弃，因为筛选已经移动到csv中
        #if is_monomer is not None:
        #    valid_mask &= (X[:, monomer_col_idx] == int(is_monomer)) # 如果是单体，则筛选出单体样本，否则筛选出非单体样本

        # 4.2 过滤无效标签(mask是按位与操作)
        valid_mask &= (y_raw != -1)   # 添加筛选条件，若为-1则过滤
        valid_mask &= ~np.isnan(y_raw) # 添加筛选条件，如果是nan则过滤掉（~为numpy中的取反操作）
        valid_mask &= (y_raw <= abnormal_point)  # 添加筛选条件，若大于异常值则过滤掉

        # 5. 应用筛选（按位置删除）
        X_valid = X[valid_mask]
        y_valid = y_raw[valid_mask]

        print(f"筛选后样本数: {X_valid.shape[0]} / {X.shape[0]}")
    ```

**4.异常情况**

无

**5.任务背景信息**

1. 当日任务：
    把去掉Morgan指纹后的特征和数据整理成一个.npy或者.pkl文件，
    然后写一个jupyter notebook展示如何读入这些预处理好的数据，方便进行特征分析和模型训练。
    然后把.npy和jupyter notebook发给我，我这边进行特征分析。（下周一前给我）

2. 几种不同的文件类型
    npyNumPy Array单数组存储。二进制格式，读写速度最快，占用空间小。存储单个大型矩阵（如特征矩阵 X）。
    npzNumPy Zipped多数组存储。本质是一个包含多个 .npy 的压缩包。同时保存 X, y, names 等多个关联变量。
    pklPickle File万能序列化。可以保存几乎任何 Python 对象（字典、模型、类）。存储复杂的嵌套字典或训练好的机器学习模型。


# 总结日志（next）