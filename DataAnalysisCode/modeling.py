#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/15
# @Author : AlexZ33
# @Site : 挖掘建模
# @File : modeling.py
# @Software: PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

"""
programmer_1-->使用随机森林算出有效特征，使用线性回归计算相关系数
programmer_2-->使用决策数模型，生成决策树过程并保存为dot文件，天气、周末、促销决定销量
programmer_3-->使用Keras神经网络模型，训练数据预测销量高低
cm_plot-->自定义混淆矩阵可视化
density_plot-->自定义概率密度图函数
programmer_4-->使用KMeans聚类，做可视化操作（概率密度图）
programmer_5-->继programmer_4将数据做降维处理，并且可视化不同聚类的类别
programmer_6-->进行白噪声、平稳性检测，建立ARIMA(0, 1, 1)模型预测之后五天的结果
programmer_7-path = os.getcwd()->使用Kmeans聚类之后，画出散点图，标记离群点
find_rule-->寻找关联规则的函数
connect_string-->自定义连接函数，用于实现L_{k-1}到C_k的连接
programmer_8-->菜单中各个菜品的关联程度
"""
path = os.getcwd()

"""
逻辑回归
自动建模
"""


def programmer_1():
    filename = path + "/data/bankloan.xls"
    data = pd.read_excel(filename)

    # x = data.iloc[:, :8].as_matrix()
    x = data.iloc[:, :8].values  # 提取所有行，０到８列
    y = data.iloc[:, 8].values
    # print(data.iloc[:, :8])
    # print(x)
    # print(y)

    rlr = RLR()  # 建立随机逻辑回归模型,筛选变量
    rlr.fit(x, y)  # 训练模型

    rlr_support = rlr.get_support(indices=True)  # 获取特征筛选结果,也可以通过.scores 方法获取各个特征的分数
    """
    这里rlr.get_support(indices=True)如果没有indices=True会报错
    机器学习之逻辑回归错误总结: https://blog.csdn.net/sheep8521/article/details/85105221
    """

    # x = data[data.columns[rlr_support]].as_matrix() # 筛选好特征
    x = data[data.columns[rlr_support]].values  # 筛选好特征
    print(u' 有效特征为:%s' % ','.join(data[data.columns[rlr_support]]))
    # 有效特征为:工龄, 地址, 负债率, 信用卡负债
    lr = LR()  # 建立逻辑回归模型
    lr.fit(x, y)  # 用筛选后的特征数据来训练模型
    print("lr: {score}".format(score=lr.score(x, y)))
    print(u' 模型的平均正确率为:% s' % lr.score(x, y))

    # 　结果　lr: 0.814285714286


if __name__ == "__main__":
    programmer_1()
    pass
