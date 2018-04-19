#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/2/26 下午12:52
# @Author  : AlexZ33
# @File    :
# @Desc    :
# @Reference: https://blog.csdn.net/muyimo/article/details/79376253

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D


# 解决中文乱码问题
myfont = fm.FontProperties(fname='/home/zk/study/python_lessions/DataVisualiztion/Library/Fonts/simsun.ttc', size=14)
matplotlib.rcParams["axes.unicode_minus"] = False

def simple_plot():
    """
    simple plot
    :return:
    """
    #生成画布
    plt.figure(figsize=(8,6), dpi=80)

    #打开交互模式
    plt.ion()

    # 循环
    for index in range(100):
        #清除原有图像
        plt.cla()
        #设定标题等
        plt.title("动态曲线图",fontproperties=myfont)
        plt.grid(True)

        #生成测试数据
        x = np.linspace(-np.pi + index, np.pi+0.1*index, 256,endpoint=True)

        y_cos, y_sin = np.cos(x),np.sin(x)

        # 设置ｘ轴
        plt.xlabel("x轴", fontproperties= myfont)
        plt.xlim(-4 + 0.1*index,4 + 0.1*index)
        plt.xticks(np.linspace(-4 + 0.1 * index, 4 + 0.1 * index, 9, endpoint=True))
        # 设置Y轴
        plt.ylabel("Y轴", fontproperties=myfont)
        plt.ylim(-1.0, 1.0)
        plt.yticks(np.linspace(-1, 1, 9, endpoint=True))

        # 画两条曲线
        plt.plot(x, y_cos, "b--", linewidth=2.0, label="cos示例")
        plt.plot(x, y_sin, "g-", linewidth=2.0, label="sin示例")


        # 设置图例位置,loc可以为[upper, lower, left, right, center]
        plt.legend(loc="upper left", prop=myfont, shadow=True)

        # 暂停
        plt.pause(0.1)

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()
    return


if __name__ == '__main__':
    simple_plot()



