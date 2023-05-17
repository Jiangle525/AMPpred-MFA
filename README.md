目录结构

```
├── 2.1 数据集信息.ipynb
├── 3.1 注意力可视化.ipynb
├── 3.2 特征可视化.ipynb
├── 3.3 消融实验.ipynb
├── AMPpred_MFA
├── dataset
├── README.md
├── multiple_training.py
├── trained_model
└── training_and_testing.py
```

- `AMPpred_MFA`：Python包，存放`AMPpred-MFA`相关的代码
- `dataset`：数据集
- `multiple_training.py`：指定正样本和负样本，构建训练集和测试集进行多次训练
- `training_and_testing.py`：指定训练集和测试集，单次训练
- `2.1 数据集信息.ipynb`：分析数据集信息
- `3.1 注意力可视化.ipynb`：`UniProt entry`为`A0A1P8AQ95`的注意力可视化，包括生成注意力矩阵热图、注意力特征排序、注意力网络等
- `3.2 特征可视化.ipynb`：分析训练集中前3000个样本的特征提取过程
- `3.3 消融实验.ipynb`：注意力的消融实验和k-mer实验