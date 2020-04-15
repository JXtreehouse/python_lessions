import pandas as pd
import numpy as np
# df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
"""
方法一
"""
print([[4,9]] * 3)
df = pd.DataFrame([[4,9]] * 3, columns=['A', 'B'])
print(df)

df1 = df.apply(np.sqrt)
print(df1)


"""
axis
0或'index'
1或“列”
"""
df2 = df.apply(np.sum, axis=0)
print("计算ａ，ｂ的和")
print(df2)
print("计算０，１，２的和")
df3 = df.apply(np.sum, axis=1)
print(df3)
df4 = df.apply(sum, axis=1)
print(df4)
df5 = df.apply(lambda x: [1, 2], axis=1)
print(df5)

df6 = df.apply(lambda x: [1, 2], axis=1, result_type='expand')
print(df6)

df7 = df.apply(lambda x: [1, 2], axis=1, result_type='reduce')
print(df7)
df8 = df.apply(lambda x: [1, 2], axis=1, result_type='broadcast')
print(df8)

df9 = df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)
print(df9)
