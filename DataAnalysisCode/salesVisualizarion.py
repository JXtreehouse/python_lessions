
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



# 读取excel数据文件
df = pd.read_excel("sample-salesv3.xlsx")
df.head()

# 查看数据总体信息
# 总共1500行数据
# df.info()

# 查看与之交易的公司数量， 共20个公司
# print(len(df.name.unique()))