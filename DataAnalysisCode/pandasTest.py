
import numpy as np
import pandas as pd

def main():

    #Data Structure
    s = pd.Series([i*2 for i in range(1,11)])
    print(type(s))

    dates = pd.date_range("20170301",periods=8)
    df = pd.DataFrame(np.random.randn(8,5),index=dates,columns=list("ABCDE"))
    print(df)
    # basic

    print(df.head(3))
    print(df.tail(3))
    print(df.index)
    print(df.values)
    print(df.T)
    # print(df.sort(columns="C"))
    print(df.sort_index(axis=1,ascending=False))
    print(df.describe())

    #select
    print(type(df["A"]))
    print(df[:3])
    print(df["20170301":"20170304"])
    print(df.loc[dates[0]])
    print(df.loc["20170301":"20170304",["B","D"]])
    print(df.at[dates[0],"C"])


    print(df.iloc[1:3,2:4])
    print(df.iloc[1,4])
    print(df.iat[1,4])

    print(df[df.B>0][df.A<0])
    print(df[df>0])
    print(df[df["E"].isin([1,2])])

    # Set
    s1 = pd.Series(list(range(10,18)),index = pd.date_range("20170301",periods=8))
    df["F"]= s1
    print(df)
    df.at[dates[0],"A"] = 0
    print(df)
    df.iat[1,1] = 1
    df.loc[:,"D"] = np.array([4]*len(df))
    print(df)

    df2 = df.copy()
    df2[df2>0] = -df2
    print(df2)

    # Missing Value
    df1 = df.reindex(index=dates[:4],columns = list("ABCD") + ["G"])
    df1.loc[dates[0]:dates[1],"G"]=1
    print(df1)
    print(df1.dropna())
    print(df1.fillna(value=1))

    # Statistic
    print(df.mean())
    print(df.var())

    s = pd.Series([1,2,4,np.nan,5,7,9,10],index=dates)
    print(s)
    print(s.shift(2))
    print(s.diff())
    print(s.value_counts())
    print(df.apply(np.cumsum))
    print(df.apply(lambda x:x.max()-x.min()))

    #Concat
    pieces = [df[:3],df[-3:]]
    print(pd.concat(pieces))

    left = pd.DataFrame({"key":["x","y"],"value":[1,2]})
    right = pd.DataFrame({"key":["x","z"],"value":[3,4]})
    print('LEFT',left)
    print('RIGHT', right)
    print(pd.merge(left,right,on="key",how="outer"))
    df3 = pd.DataFrame({"A": ["a","b","c","b"],"B":list(range(4))})
    print(df3.groupby("A").sum())

    # DataFrame合并数据集
    # 这是一种多对一合并
    df1 = pd.DataFrame({
      'key': ['b','b','a','c','a','a','b'],
      'data1': range(7)
    })
    df2 = pd.DataFrame({
        'key': ['a','b','d'],
        'data2': range(3)
    })
    print(pd.merge(df1, df2))

    # 注意： 我们没有指明用哪个列进行连接，默认会将重叠列的列名当做键
    # 最好显示指定下
    print(pd.merge(df1,df2,on='key'))


    ###
    #索引上的合并
    # 有时候，DataFrame中的连接键位于其索引中，这种情况下，你可以传入left_index=True 或right_index=True以说明索引应该被用作连接键
    ###
    left1 = pd.DataFrame({
        'key': ['a','b','a','a','b','c'],
        'value': range(6)
        })
    right1 = pd.DataFrame({
        'group_val':[3.5, 7]
        },index= ['a', 'b'])
    print(left1)
    print(right1)
    print(pd.merge(left1, right1, left_on='key', right_index=True))
    # 因为默认的merge方法是求取连接键的交集，因此你可以通过外连接的方式得到它们的并集
    print(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))



   # left2 = pd.DataFrame({
   #     'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
   #     'key2': [2000, 2001, 2002, 2001, 2002],
   #     'data': np.arrange(5.)
   # })
   #
   # right2 = pd.DataFrame(np.arrange(12).reshape((6, 2)),
   #                    index=[['Nevada', 'Nevada', 'Ohio','Ohio', 'Ohio', 'Ohio'],
   #                    [2001, 2000, 2000, 2000, 2001, 2002]],
   #                    columns = ['event1', 'event2'])
   # print(left2)
if __name__ == "__main__":
    main()
