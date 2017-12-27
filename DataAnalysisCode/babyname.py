
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
# plt.show()

###
#起名趋势分析
###
#增加一个变量rank，这个是根据年份性别依据名字出现频率所产生的次序
baby_names['ranked'] = baby_names.groupby(['year','gender'])['frequency'].rank(ascending=False)
# print(baby_names.head(10))

#计算每个名每年按性别占总出生人数的百分比
def add_pct(group):#自定义
    group['pct'] = group.frequency / group.frequency.sum()*100
    return group
# #groupby和apply函数
baby_names = baby_names.groupby(['year','gender']).apply(add_pct)
# # 查看新加的百分比（pct）
# print(baby_names.head())

####
#查看每年最流行的名字所占百分比趋势
####

#将数据分为男孩和女孩
dff = baby_names[baby_names.gender == 'F']
dfm = baby_names[baby_names.gender == 'M']
#获取每年排名第一的名字
rank1m = dfm[dfm.ranked == 1]
rank1f = dff[dff.ranked == 1]

plt.plot(rank1m.year, rank1m.pct, color="blue", linewidth = 2, label = 'Boys')
plt.fill_between(rank1m.year, rank1m.pct, color="blue", alpha = 0.1, interpolate=True)
plt.xlim(1880,2012)
plt.ylim(0,9)
plt.xticks(scipy.arange(1880,2012,10), rotation=70)
plt.title("Popularity of #1 boys' name by year", size=18, color="blue")
plt.xlabel('Year', size=15)
plt.ylabel('% of male births', size=15)
plt.show()
plt.close()


plt.plot(rank1f.year, rank1f.pct, color="red", linewidth = 2, label = 'Girls')
plt.fill_between(rank1f.year, rank1f.pct, color="red", alpha = 0.1, interpolate=True)
plt.xlim(1880,2012)
plt.ylim(0,9)
plt.xticks(scipy.arange(1880,2012,10), rotation=70)
plt.title("Popularity of #1 girls' name by year", size=18, color="red")
plt.xlabel('Year', size=15)
plt.ylabel('% of female births', size=15)
plt.show()
plt.close()