
# 首先产生一个叫gdp的字典
gdp = {"country":["United States", "China", "Japan", "Germany", "United Kingdom"],
       "capital":["Washington, D.C.", "Beijing", "Tokyo", "Berlin", "London"],
       "population":[323, 1389, 127, 83, 66],
       "gdp":[19.42, 11.8, 4.84, 3.42, 2.5],
       "continent":["North America", "Asia", "Asia", "Europe", "Europe"]}

import pandas as pd
gdp_df = pd.DataFrame(gdp)
print(gdp_df)

# 我们可以通过index选项添加自定义的行标签(label)
# 使用column选项可以选择列的顺序
gdp_df = pd.DataFrame(gdp, columns = ["country", "capital", "population", "gdp", "continent"],index = ["us", "cn", "jp", "de", "uk"])
print(gdp_df)

#修改行和列的标签
# 也可以使用index和columns直接修改
gdp_df.index=["US", "CN", "JP", "DE", "UK"]
gdp_df.columns = ["Country", "Capital", "Population", "GDP", "Continent"]
print(gdp_df)
# 增加rank列，表示他们的GDP处在前5位
gdp_df["rank"] = "Top5 GDP"
# 增加国土面积变量,以百万公里计（数据来源：http://data.worldbank.org/）
gdp_df["Area"] = [9.15, 9.38, 0.37, 0.35, 0.24]
print(gdp_df)


# 一个最简单的series
series = pd.Series([2,4,5,7,3],index = ['a','b','c','d','e'])
print(series)
# 当我们使用点操作符来查看一个变量时，返回的是一个pandas series
# 在后续的布尔筛选中使用点方法可以简化代码
# US,...,UK是索引
print(gdp_df.GDP)


# 可以直接查看索引index
print(gdp_df.GDP.index)
# 类型是pandas.core.series.Series
print(type(gdp_df.GDP))

#返回一个布尔型的series，在后面讲到的DataFrame的布尔索引中会大量使用
print(gdp_df.GDP > 4)

# 我们也可以将series视为一个长度固定且有顺序的字典，一些用于字典的函数也可以用于series
gdp_dict = {"US": 19.42, "CN": 11.80, "JP": 4.84, "DE": 3.42, "UK": 2.5}
gdp_series = pd.Series(gdp_dict)
print(gdp_series)

# 判断 ’US' 标签是否在gdp_series中

print("US" in gdp_series)
# 使用变量名加[[]]选取列
print(gdp_df[["Country"]])
# 可以同时选取多列
print(gdp_df[["Country", "GDP"]])


# 如果只是用[]则产生series
print(type(gdp_df["Country"]))
# 行选取和2d数组类似
# 如果使用[]选取行，切片方法唯一的选项
print(gdp_df[2:5]) #终索引是不被包括的！

#loc方法
# 在上面例子中，我们使用行索引选取行，能不能使用行标签实现选取呢？
# loc方法正是基于标签选取数据的方法
print(gdp_df.loc[["JP","DE"]])
# 以上例子选取了所有的列
# 我们可以加入需要的列标签
print(gdp_df.loc[["JP","DE"],["Country","GDP","Continent"]])

# 选取所有的行，我们可以使用:来表示选取所有的行
print(gdp_df.loc[:,["Country","GDP","Continent"]])

# 等价于gdp_df.loc[["JP","DE"]]
print(gdp_df.iloc[[2,3]])

print(gdp_df.loc[["JP","DE"],["Country", "GDP", "Continent"]])
print(gdp_df.iloc[[2,3],[0,3,4]])

# 选出亚洲国家，下面两行命令产生一样的结果
print(gdp_df[gdp_df.Continent == "Asia"])

print(gdp_df.loc[gdp_df.Continent == "Asia"])
# 选出gdp大于3兆亿美元的欧洲国家
print(gdp_df[(gdp_df.Continent == "Europe") & (gdp_df.GDP > 3)])