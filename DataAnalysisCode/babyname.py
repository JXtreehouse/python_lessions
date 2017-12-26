
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re as re
import scipy


#读取数据


years = range(1880, 2017) # range函数
pieces = [] # 列表
columns = ['name', 'gender', 'frequency']

for year in years: # for循环
    path = './names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns) # 数据读取
    frame['year'] = year # 增加变量year
    pieces.append(frame)

baby_names = pd.concat(pieces, ignore_index=True) # 转换为pd数据： 第五节课讲解

# baby_names.head(10)

#
# print(baby_names.describe())
# print(baby_names)
# print(head)
# 按照名字将数据分组,总数，平均数，标准差
#print(baby_names.groupby('name').agg([np.sum,np.mean,np.std]))

###
# 哪些名字出现的频率最高？
####
#print(baby_names.groupby('name').agg({'frequency': sum}))


# James, John, Robert, Micheal, Mary...都是耳熟能详的名字
#print(baby_names.groupby('name').agg({'frequency': sum}).sort_values(by=['frequency'], ascending=[0]))


####
#每年出生的男孩和女孩的个数分别是多少？
####


# 使用pivot_table方法查看
freq_by_gender_year = baby_names.pivot_table(index = 'year',columns='gender',values='frequency',aggfunc=sum)
#print(freq_by_gender_year)

# 使用tail方法查看最近几年出生人数
#print(freq_by_gender_year.tail())

# 一行命令即可做出高质量图形
freq_by_gender_year.plot(title='Frequency by year and gender')
plt.show()

