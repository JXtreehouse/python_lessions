
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



if __name__ == "__main__":
    main()
