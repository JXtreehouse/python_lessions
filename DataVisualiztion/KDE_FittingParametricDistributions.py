# Fitting parametric distributions拟合参数分布

# 可以利用distplot() 把数据拟合成参数分布的图形并且观察它们之间的差距,再运用fit来进行参数控制。


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

#引入鸢尾花数据集
df_iris = sns.load_dataset("iris")

# Plot a simple histogram with binsize determined automatically

sns.distplot(df_iris['petal_length'], ax= axes[0, 0], kde = False, color="b")

# Plot a kernel density estimate and rug plot

sns.distplot(df_iris['petal_length'], ax= axes[0, 1], kde = False, color="r", rug=True)

# Plot a filled kernel density estimate

sns.distplot(df_iris['petal_length'], ax= axes[1, 0], hist = False, color="g", kde_kws={"shade": True})

# Plot a historgram and kernel density estimate
sns.distplot(df_iris['petal_length'], color="m", ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()





