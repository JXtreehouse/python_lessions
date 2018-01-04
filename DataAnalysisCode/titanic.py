# 读取常用的包
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#读取数据
titanic_df = pd.read_csv('data/titanic.csv')

#查看前五行数据
print(titanic_df.head())

# 数据的统计描述
# describe函数查看部分变量的分布
# 因为Survived是0-1变量，所以均值就是幸存人数的百分比，这个用法非常有用
print(titanic_df[["Survived","Age","SibSp","Parch"]].describe())

# 使用include=[np.object]来查看分类变量
# count: 非缺失值的个数
# unique: 非重复值得个数
# top: 最高频值
# freq: 最高频值出现次数

print(titanic_df.describe(include=[np.object]))

#不同舱位的分布情况是怎样的呢？
# 方法1: value_counts
# 查看不同舱位的分布
# 头等舱：24%； 二等舱：21%； 三等舱：55%
# value_counts 频数统计， len() 获取数据长度
print(titanic_df.Pclass.value_counts() / len(titanic_df))
# 总共有891个乘客
# Age有714个非缺失值，Cabin只有204个非缺失值。我们将会讲解如何处理缺失值
print(titanic_df.info())

#方法2：group_by
# sort_values 将结果排序
(titanic_df.groupby("Pclass").agg("size")/len(titanic_df)).sort_values(ascending=False)

# 填补年龄数据中的缺失值
# 直接使用所有人年龄的中位数来填补
# 在处理之前，查看Age列的统计值
print(titanic_df.Age.describe())

# 重新载入原始数据
titanic_df=pd.read_csv("data/titanic.csv")

# 计算所有人年龄的均值
age_median1 = titanic_df.Age.median()

# 使用fillna填充缺失值,inplace=True表示在原数据titanic_df上直接进行修改
titanic_df.Age.fillna(age_median1,inplace=True)
#查看Age列的统计值
print(titanic_df.Age.describe())
#print(titanic_df.info())

# 考虑性别因素，分别用男女乘客各自年龄的中位数来填补
# 重新载入原始数据
titanic_df=pd.read_csv("data/titanic.csv")
# 分组计算男女年龄的中位数， 得到一个Series数据，索引为Sex
age_median2 = titanic_df.groupby('Sex').Age.median()
# 设置Sex为索引
titanic_df.set_index('Sex',inplace=True)
# 使用fillna填充缺失值，根据索引值填充
titanic_df.Age.fillna(age_median2, inplace=True)
# 重置索引，即取消Sex索引
titanic_df.reset_index(inplace=True)
# 查看Age列的统计值
print(titanic_df.Age.describe())

#同时考虑性别和舱位因素

# 重新载入原始数据
titanic_df=pd.read_csv("data/titanic.csv")
# 分组计算不同舱位男女年龄的中位数， 得到一个Series数据，索引为Pclass,Sex
age_median3 = titanic_df.groupby(['Pclass', 'Sex']).Age.median()
# 设置Pclass, Sex为索引， inplace=True表示在原数据titanic_df上直接进行修改
titanic_df.set_index(['Pclass','Sex'], inplace=True)
print(titanic_df)

# 使用fillna填充缺失值，根据索引值填充
titanic_df.Age.fillna(age_median3, inplace=True)
# 重置索引，即取消Pclass,Sex索引
titanic_df.reset_index(inplace=True)

# 查看Age列的统计值
titanic_df.Age.describe()


###
#分析哪些因素会决定生还概率
###

# 舱位与生还概率
#计算每个舱位的生还概率
# 方法1：使用经典的分组-聚合-计算
# 注意：因为Survived是0-1函数，所以均值即表示生还百分比
print(titanic_df[['Pclass', 'Survived']].groupby('Pclass').mean() \
    .sort_values(by='Survived', ascending=False))

# 方法2：我们还可以使用pivot_table函数来实现同样的功能（本次课新内容）
# pivot table中文为数据透视表
# values: 聚合后被施加计算的值，这里我们施加mean函数
# index: 分组用的变量
# aggfunc: 定义施加的函数
print(titanic_df.pivot_table(values='Survived', index='Pclass', aggfunc=np.mean))

# 绘制舱位和生还概率的条形图
# 使用sns.barplot做条形图，图中y轴给出 Survived 均值的点估计
#sns.barplot(data=titanic_df,x='Pclass',y='Survived',ci=None)
# plt.show()

#####
#性别与生还概率
#####
# 方法1：groupby
print(titanic_df[["Sex", "Survived"]].groupby('Sex').mean() \
    .sort_values(by='Survived', ascending=False))
# 方法2：pivot_table
print(titanic_df.pivot_table(values="Survived",index='Sex',aggfunc=np.mean))

# 绘制条形图
#sns.barplot(data=titanic_df,x='Sex',y='Survived',ci=None)
#plt.show()


#####
#综合考虑舱位和性别的因素，与生还概率的关系
#####
# 方法1：groupby
print(titanic_df[['Pclass','Sex', 'Survived']].groupby(['Pclass', 'Sex']).mean())

# 方法2：pivot_table
titanic_df.pivot_table(values='Survived', index=['Pclass', 'Sex'], aggfunc=np.mean)

# 方法3：pivot_talbe
# columns指定另一个分类变量，只不过我们将它列在列里而不是行里，这也是为什么这个变量称为columns
print(titanic_df.pivot_table(values="Survived",index="Pclass",columns="Sex",aggfunc=np.mean))

#绘制条形图：使用sns.barplot
#sns.barplot(data=titanic_df,x='Pclass',y='Survived',hue='Sex',ci=None)
# plt.show()

# 绘制折线图：使用sns.pointplot
sns.pointplot(data=titanic_df,x='Pclass',y="Survived",hue="Sex",ci=None)
#plt.show()

####
#年龄与生还情况
####
#与上面的舱位、性别这些分类变量不同，年龄是一个连续的变量

#生还组和罹难组的年龄分布直方图
#使用seaborn包中的 FacetGrid().map() 来快速生成高质量图片
# col='Survived'指定将图片在一行中做出生还和罹难与年龄的关系图
sns.FacetGrid(titanic_df,col='Survived').\
    map(plt.hist,'Age',bins=20,normed=True)
# plt.show()


###
#将连续型变量离散化
###
#我们使用cut函数
#我们可以看到每个区间的大小是固定的,大约是16岁

titanic_df['AgeBand'] = pd.cut(titanic_df['Age'],5)
print(titanic_df.head())

#查看落在不同年龄区间里的人数
#方法1：value_counts(), sort=False表示不需要将结果排序
print(titanic_df.AgeBand.value_counts(sort=False))

#方法2：pivot_table
print(titanic_df.pivot_table(values='Survived',index='AgeBand',aggfunc='count'))

#查看各个年龄区间的生还率
print(titanic_df.pivot_table(values="Survived",index='AgeBand',aggfunc=np.mean))
sns.barplot(data=titanic_df,x='AgeBand',y='Survived',ci=None)
plt.xticks(rotation=60)
plt.show()


####
# 年龄、性别 与生还概率
####
# 查看落在不同区间里男女的生还概率
print(titanic_df.pivot_table(values='Survived',index='AgeBand', columns='Sex', aggfunc=np.mean))

sns.pointplot(data=titanic_df, x='AgeBand', y='Survived', hue='Sex', ci=None)
plt.xticks(rotation=60)

plt.show()

####
#年龄、舱位、性别 与生还概率
####
titanic_df.pivot_table(values='Survived',index='AgeBand', columns=['Sex', 'Pclass'], aggfunc=np.mean)



# 回顾sns.pointplot 绘制舱位、性别与生还概率的关系图
sns.pointplot(data=titanic_df, x='Pclass', y='Survived', hue='Sex', ci=None)