# 20251225工作内容存档

**1.特殊注意**
* 该文件夹中的所有文件本应属于上一级目录/notebook/
* 如果需要重新使用，请将py文件移动到上一层文件夹中
* 20251220-20251225：消融实验：验证去掉Morgan或者Avalon分子指纹后（或者全部去除），平均的**性能**/**数据稳定性**以及**迁移效果**的变化
* 在某一页ppt中，迁移学习的效果，原本是All但是title上写了Morgan，这里是笔误，忽略不计即可，后期有机会


**2.简要总结**

本文件夹用于存放 2025-12-25 期间的 Ablation（消融）实验相关资料。包含用于清洗与汇总交叉验证结果的脚本（如 `clean_cv_json.py`、`aggregate_cv_json.py`）、用于生成对比图和可视化的脚本（如 `generate_cv_figure.py`），以及若干用于复现与稳定运行的实验脚本（如 `Phase3_1_stable.py`、`Phase3_2_stable.py`、`scr.py`）。目录还包含实验产生的 JSON 数据和图像目录（`cv_figure/`、`trans_figures/`），总体目的是集中管理消融实验的数据、脚本与图表，方便结果复现、对比与可视化分析。

**3.工作流程-->**

**检查5折交叉验证并且获取结果**
* Phase3_1_stable.py生成data.json(记录每一次的交叉验证结果)
* clean_cv_json.py生成data_cleaned.json(清除null值和NaN)
* aggregate_cv_json.py生成data_average.json(计算100次的平魂之)
* generate_cv_figure.py，根据四次平均值计算的情况，计算热力图

**迁移学习效果比较**
* 直接运行Phase3_2_stable.py直接生成对应的迁移学习热力图
* 如果需要检查消融（即同一个数据位置，四种消融的区别），可以使用scr.py绘制折线图

**4.工作内容总结**
* 报告保存在pptx文件中，在该目录下
* pdf文件同步留档
