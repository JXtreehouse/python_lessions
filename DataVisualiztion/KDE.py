import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#引入鸢尾花数据集
df_iris = sns.load_dataset("iris")
fig, axes = plt.subplots(1,2)
# print(df_iris['petal_length'])
# print(axes[0])

# distplot 函数默认同时绘制直方图和KDE(核密度图),开启rug细条
sns.distplot(df_iris['petal_length'], ax= axes[0], hist = False, rug = True)
# shade 阴影
sns.kdeplot(df_iris['petal_length'], ax = axes[1], shade = True)

plt.show()