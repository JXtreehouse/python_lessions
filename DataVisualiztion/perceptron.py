#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/7 下午7:35
# @Author : AlexZ33
# @Site : 实现感知器对象
# @reference: https://www.imooc.com/video/14378
# @File : perceptron.py.py
# @Software: PyCharm

import numpy as np
class Perceptron (object):
    """
    eta: 学习率
    n_iter: 权重向量的训练次数
    w_: 神经分叉权重向量
    errors: 用于记录神经元判断出错次数
    """
    # 初始化
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter;

        pass
    # 根据神经元样本进行培训

    def fit(self, x, y):
        """
        输入训练数据,培训神经元
           x: 输入样本向量(电信号向量)
           y: 对应的样本分类
         X: shape[n_samples, n_features]
         n_samples输入的样本量
         n_features输入的分叉总共有多少
         eg:
         X: [[1,2,3], [4,5,6]]
         n_samples: 2
         n_features: 3

         y: [1, -1]
        """
        # 初始化权重向量为0
        self.w_ = np.zero(1 + X.shape[1])
        self.errrors_ = [];

        for _ in range(self.n_iter):
            errors = 0
            """
            x: [[1,2,3],[4,5,6]]
            y: [1, -1]
            zip(x,y) = [[1,2,3,1], [4,5,6,-1]]
            """
            for xi, target in zip(x, y):
                update = self.eta * (target- self.predict(xi))
        pass