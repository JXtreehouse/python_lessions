
import numpy as np
#  读取文件
# 读取红酒品质数据，原数据以分号进行分隔，并且跳过第一行
# 数据文件中第一行是字段名，属于字符串类型，而numpy数组的所有元素必须是一致的数据类型，故需要跳过第一行，否则会出错。
wine_data = np.genfromtxt('data/winequality-red.csv',delimiter=';',skip_header=1)
print(wine_data)

# 查看数组的形状，行和列
print(wine_data.shape)

print(wine_data.dtype)
# 取前10行数据中的8，10，11列,分别对应红酒的PH值、酒精度、质量评分这三类属性
# 为简单起见，后续将使用这组数据进行分析示例
wine = wine_data[:10,[8,10,11]]
print(wine)
print(wine[:,2] > 5)

# 常用函数
# 首先用np.sum()获取数据的和
# 这里将数组中的所有元素相加了，但是我们想要的是每一列数据的和
print(np.sum(wine))

# 使用axis参数来设置求和的方式， axis=0表示对列求和，axis=1表示对行求和
print(np.sum(wine,axis=0))

#每列的总和除以行数，得到每列的均值
print(np.sum(wine,axis=0) / len(wine))

# 直接使用np.mean()函数，但记得设置axis参数
print(np.mean(wine, axis=0))



