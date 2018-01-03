#####
#数据的导入和观察
#####
import pandas as pd
# 用列表存储列标签
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# 读取数据，并指定每一列的标签
iris = pd.read_csv('data/iris.txt', names = col_names)

# 使用head/tail查看数据的头和尾

print(iris.head(10))

# 使用info 方法查看数据的总体信息
iris.info()

# 使用shape可以查看DataFrame的行数与列数
# iris有150个观察值，5个变量
print(iris.shape)
# 这里的品种(species)是分类变量(categorical variable)
# 可以使用unique方法来对查看series中品种的名字
print(iris.species.unique())


# 统计不同品种的数量
# 使用DataFrame的value_counts方法来实现
print(iris.species.value_counts())

#选取花瓣数据，即 petal_length 和 petal_width 这两列
# 方法一：使用[[ ]]
petal = iris[['petal_length','petal_width']]
print(petal.head())
# 方法二：使用 .loc[ ]
petal = iris.loc[:,['petal_length','petal_width']]
print(petal.head())
# 方法三：使用 .iloc[ ]
petal = iris.iloc[:,2:4]
print(petal.head())

# 选取行索引为5-10的数据行
# 方法一：使用[]
print(iris[5:11])
# 方法二：使用 .iloc[]
print(iris.iloc[5:11,:])

# 选取品种为 Iris-versicolor 的数据
versicolor = iris[iris.species == 'Iris-versicolor']
print(versicolor.head())


####
#数据的可视化
####
#散点图
import matplotlib.pyplot as plt
# 我们首先画散点图（sactter plot），x轴上画出花瓣的长度，y轴上画出花瓣的宽度
# 我们观察到什么呢？
iris.plot(kind = 'scatter', x="petal_length", y="petal_width")
# plt.show()

# 使用布尔索引的方法分别获取三个品种的数据
setosa = iris[iris.species == 'Iris-setosa']
versicolor = iris[iris.species == 'Iris-versicolor']
virginica = iris[iris.species == 'Iris-virginica']

ax = setosa.plot(kind='scatter', x="petal_length", y="petal_width", color='Red', label='setosa', figsize=(10,6))
versicolor.plot(kind='scatter', x="petal_length", y="petal_width", color='Green', ax=ax, label='versicolor')
virginica.plot(kind='scatter', x="petal_length", y="petal_width", color='Orange', ax=ax, label='virginica')
plt.show()

#箱图
#使用mean()方法获取花瓣宽度均值
print(iris.petal_width.mean())
#使用median()方法获取花瓣宽度的中位数
print(iris.petal_width.median())
# 可以使用describe方法来总结数值变量
print(iris.describe())


# 绘制花瓣宽度的箱图
# 箱图展示了数据中的中位数，四分位数，最大值，最小值
iris.petal_width.plot(kind='box')
# plt.show()

# 按品种分类，分别绘制不同品种花瓣宽度的箱图
iris[['petal_width','species']].boxplot(grid=False,by='species',figsize=(10,6))
# plt.show()

setosa.describe()

# 计算每个品种鸢尾花各个属性（花萼、花瓣的长度和宽度）的最小值、平均值又是分别是多少？ （提示：使用min、mean 方法。）
print(iris.groupby(['species']).agg(['min','mean']))

#计算鸢尾花每个品种的花萼长度（sepal_length) 大于6cm的数据个数。
# 方法1
print(iris[iris['sepal_length']> 6].groupby('species').size())
# 方法2
def more_len(group,length=6):
    return len(group[group['sepal_length'] > length])
print(iris.groupby(['species']).apply(more_len,6))