
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re as re
import scipy

#读取数据


years = range(1880, 2017) # range函数：第三节课讲解
pieces = [] # 列表：第二节课讲解
columns = ['name', 'gender', 'frequency']

for year in years: # for循环：第二节课讲解
    path = './../names/yob%d.txt'
    frame = pd.read_csv(path, names=columns) # 数据读取： 第八节课讲解
    frame['year'] = year # 增加变量year：第八节课讲解
    pieces.append(frame)
baby_names = pd.concat(pieces, ignore_index=True) # 转换为pd数据： 第五节课讲解
baby_names
baby_names.head(10)

baby_names.describe()
