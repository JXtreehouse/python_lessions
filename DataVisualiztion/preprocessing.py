#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/1 下午2:32
# @Author : AlexZ33
# @Site :  数据预处理
# @document: https://blog.csdn.net/qq_43333395/article/details/89330504
# https://blog.csdn.net/sinat_25873421/article/details/80634976
# @File : preprocessing.py
# @Software: PyCharm

import os
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# 主要用于小波变换包含单层（逆）变换、多尺度变换和阈值函数等
import pywt
from pandas import DataFrame, Series
# 导入拉格朗日插值函数
from scipy.interpolate import lagrange
from scipy.io import loadmat  # mat是MATLAB的专用格式，调用loadmat方法读取
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""
代码说明：
ployinterp_column-->拉格朗日填充数值
programmer_1-->筛选异常数据（包括NaN）进行填充
programmer_2-->最小-最大规范化、零-均值规范化、小数定标规范化
programmer_4-->基本的dataframe操作
programmer_5-->利用小波分析（？？？）进行特征分析
programmer_6-->利用PCA计算特征向量，用于降维分析
"""

path = os.getcwd()


def mkdir(cpath):
    # 引入模块
    import os
    import shutil

    # 去除首位空格
    path = cpath.strip()
    # 去除尾部 \ 符号
    path = cpath.rstrip("//")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print cpath + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print cpath + ' 目录已存在'
        shutil.rmtree(cpath)
        mkdir(cpath)


def programmer_1():
    # 销量数据路径
    inputfile = path + '/data/catering_sale.xls'
    # 输出数据路径
    outputfile = path + '/tmp/sales.xls'
    outputfiledir = path + '/tmp/'
    data = pd.read_excel(inputfile)

    data[(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None  # 过滤异常值，将其变为空值
    print(data)

    # 自定义列向量插值函数
    # https: // www.jianshu.com / p / bf6803447dce
    # df为列向量，index为被插值的位置，k为取前后的数据个数，默认为5
    def ployinterp_column(index, df, k=5):
        # type: (int, dataframe, int) -> object
        y = df[list(range(index - k, index))
               + list(range(index + 1, index + 1 + k))]  # 当index=7时，取数y=df[2，3，4，5，6，8，9，10，11，12]
        y = y[y.notnull()]
        return lagrange(y.index, list(y))(index)

    df = data[data[u'销量'].isnull()]
    print('.........................')
    print(df)

    print('.........................')
    index_list = df[u'销量'].index

    assert isinstance(index_list, object)
    print(index_list)

    for index in index_list:
        data[[u'销量']][index] = ployinterp_column(index, data[u'销量'])
    mkdir(outputfiledir)
    data.to_excel(outputfile)


# 规范化
# 　为了消除指标之间的量纲和取值范围差异的影响，进行标准化处理，将数据按照比例进行缩放，使之落入一个特定的区域，便于进行综合分析。

def programmer_2():
    datafile = path + '/data/normalization_data.xls'
    data = pd.read_excel(datafile, header=None)

    print ((data - data.min()) / (data.max() - data.min()))  # 最小-最大规范化(离差标准化)
    print((data - data.mean()) / data.std())  # 零-均值规范化（标准差标准化）
    print(data / 10 * np.ceil(np.log10(data.abs().max())))


# 数据规范化
def programmer_3():
    datafile = path + '/data/discretization_data.xls'
    data = pd.read_excel(datafile)
    data = data[u'肝气郁结证型系数'].copy()
    k = 4
    # 方法一，直接对数组进行分类
    # 等宽法
    d1 = pd.cut(data, k, labels=range(k))  # 等宽离散化，各个类比依次命名为0,1,2,3
    print(d1)

    # 方法二， 等频率离散化
    w = [1.0 * i / k for i in range(k + 1)]
    # percentiles表示特定百分位数，同四分位数
    w = data.describe(percentiles=w)[4:4 + k + 1]
    w[0] = w[0] * (1 - 1e-10)
    d2 = pd.cut(data, w, labels=range(k))

    # 　方法三，使用Kmeans 基于聚类分析的方法
    kmodel = KMeans(n_clusters=k, n_jobs=4)  # 建立模型，n_jobs是并行数，一般等于CPU数较好
    # print(data.values.reshape(len(data), 1))
    kmodel.fit(data.values.reshape(len(data), 1))  # 训练模型

    # 输出聚类中心，并且排序(默认是随机序的）
    c = DataFrame(kmodel.cluster_centers_).sort_values(0)
    # 相邻两项求中点，作为边界点
    w = DataFrame.rolling(c, 2).mean().iloc[1:]

    # 加上首末边界点
    w = [0] + list(w[0]) + [data.max()]  # 把首末边界点加上，w[0]中0为列索引
    d3 = pd.cut(data, w, labels=range(k))
    print(d3)

    def cluster_plot(d, k):  # 自定义作图函数来显示聚类结果
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        plt.figure(figsize=(8, 3))
        for j in range(0, k):
            plt.plot(data[d == j], [j for i in d[d == j]], 'o')
        plt.ylim(-0.5, k - 0.5)
        return plt

    cluster_plot(d1, k).show()
    cluster_plot(d2, k).show()
    cluster_plot(d3, k).show()


if __name__ == '__main__':
    # programmer_1()
    programmer_3()
    # programmer_2()
    pass
