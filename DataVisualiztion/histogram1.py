
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt  #导入

import seaborn as sns
sns.set(color_codes=True)#导入seaborn包设定颜色

np.random.seed(sum(map(ord, "distributions")))
x = np.random.normal(size=100)
# sns.distplot(x, kde=False, rug=True);#kde=False关闭核密度分布,rug表示在x轴上每个观测上生成的小细条（边际毛毯）
sns.distplot(x, bins=20, kde=False, rug=True);#设置了20个矩形条
plt.show()