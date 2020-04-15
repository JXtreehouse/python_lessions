import pandas as pd

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
"""
方法一
"""
df['c'] = df.apply(lambda row: row.a + row.b, axis=1)
print(df)

"""
方法二
"""


def add(x):
    return x.a + x.b


df['d'] = df.apply(add, axis=1)
print(df)

"""
方法三
"""
df['e'] = df.apply(sum, axis=1)
print(df)

"""
方法四
"""
df['f'] = df.apply(add, axis=1)

print(df)

"""
方法五
"""
df.loc[len(df)] = df.apply(sum, axis=0)
print(df)
