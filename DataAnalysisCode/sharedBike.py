#租车人数是由哪些因素决定的？
#导入数据分析包
import numpy as np
import pandas as pd

#导入绘图工具包
import matplotlib.pyplot as plt
import seaborn as sns

#导入日期时间变量处理相关的工具包
import calendar
from datetime import datetime

# 读取数据
BikeData = pd.read_csv('data/bike.csv')


#####
#了解数据大小
#查看前几行/最后几行数据
#查看数据类型与缺失值
####
# 第一步：查看数据大小

print(BikeData.shape)

# 第二步：查看前10行数据
print(BikeData.head(10))


# 第三步：查看数据类型与缺失值
# 大部分变量为整数型，温度和风速为浮点型变量
# datetime类型为object，我们将在下面进一步进行处理
# 没有缺失值！
print(BikeData.info())


####
#日期型变量的处理
####

# 取datetime中的第一个元素为例，其数据类型为字符串，所以我们可以使用split方法将字符串拆开
# 日期+时间戳是一个非常常见的数据形式
ex = BikeData.datetime[1]
print(ex)

print(type(ex))

# 使用split方法将字符串拆开
ex.split()

# 获取日期数据
ex.split()[0]

# 首先获得日期，定义一个函数使用split方法将日期+时间戳拆分为日期和
def get_date(x):
    return(x.split()[0])

# 使用pandas中的apply方法，对datatime使用函数get_date
BikeData['date'] = BikeData.datetime.apply(get_date)

print(BikeData.head())

# 生成租车时间(24小时）
# 为了取小时数，我们需要进一步拆分
print(ex.split()[1])
#":"是分隔符
print(ex.split()[1].split(":")[0])

# 将上面的内容定义为get_hour的函数，然后使用apply到datatime这个特征上
def get_hour(x):
    return (x.split()[1].split(":")[0])
# 使用apply方法，获取整列数据的时间
BikeData["hour"] = BikeData.datetime.apply(get_hour)

print(BikeData.head())

####
# 生成日期对应的星期数
####
# 首先引入calendar中的day_name，列举了周一到周日
print(calendar.day_name[:])

#获取字符串形式的日期
dateString = ex.split()[0]

# 使用datatime中的strptime函数将字符串转换为日期时间类型
# 注意这里的datatime是一个包不是我们dataframe里的变量名
# 这里我们使用"%Y-%m-%d"来指定输入日期的格式是按照年月日排序，有时候可能会有月日年的排序形式
print(dateString)
dateDT = datetime.strptime(dateString,"%Y-%m-%d")
print(dateDT)
print(type(dateDT))

# 然后使用weekday方法取出日期对应的星期数
# 是0-6的整数，星期一对应0， 星期日对应6
week_day = dateDT.weekday()

print(week_day)
# 将星期数映射到其对应的名字上
print(calendar.day_name[week_day])


# 现在将上述的过程融合在一起变成一个获取星期的函数
def get_weekday(dateString):
    week_day = datetime.strptime(dateString,"%Y-%m-%d").weekday()
    return (calendar.day_name[week_day])

# 使用apply方法，获取date整列数据的星期
BikeData["weekday"] = BikeData.date.apply(get_weekday)

print(BikeData.head())


####
# 生成日期对应的月份
####

# 模仿上面的过程，我们可以提取日期对应的月份
# 注意：这里month是一个attribute不是一个函数，所以不用括号

def get_month(dateString):
    return (datetime.strptime(dateString,"%Y-%m-%d").month)
# 使用apply方法，获取date整列数据的月份
BikeData["month"] = BikeData.date.apply(get_month)
print(BikeData.head())

####
#数据可视化举例
####

#绘制租车人数的箱线图， 以及人数随时间（24小时）变化的箱线图
# 设置画布大小
fig = plt.figure(figsize=(18,5))

# 添加第一个子图
# 租车人数的箱线图
ax1 = fig.add_subplot(121)
sns.boxplot(data=BikeData,y="count")
ax1.set(ylabel="Count",title="Box Plot On Count")


# 添加第二个子图
# 租车人数和时间的箱线图
# 商业洞察：租车人数由时间是如何变化的?
ax2 = fig.add_subplot(122)
sns.boxplot(data=BikeData,y="count",x="hour")
ax2.set(xlabel="Hour",ylabel="Count",title="Box Plot On Count Across Hours")
plt.show()


##########
#相关性分析（Correlation Analysis）
##########