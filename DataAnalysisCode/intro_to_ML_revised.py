
# sklearn是机器学习模型的主要工具
import matplotlib.pyplot as plt #作图
from sklearn import datasets #数据
from sklearn.feature_selection import SelectKBest,f_regression #特征选取
from sklearn.linear_model import LinearRegression #线性回归
from sklearn import metrics #线性回归
import numpy as np
import pandas as pd


####
#读取数据
#房价预测是典型的监督学习，所以我们的训练集（training data）要求特征和目标
####
#读取波士顿房价数据
boston_dataset = datasets.load_boston()
X_full = boston_dataset.data #导入特征  # X_full是一个ndarray
Y = boston_dataset.target # 导入目标：房屋中间价 1000美元计价

print (X_full.shape) #shape method读取数据尺寸，我们有506个数据点和13个变量
print (Y.shape)  #目标一般都是一维变量。Y是一个ndarray。

#feature_names 列举了所有特征名字，上方有其对应的描述
print(boston_dataset.feature_names)

# 把数据分为训练集和测试集
# 为了测试模型的表现，我们将数据分为70%的训练集和30%的测试集
# 使用train_test_split函数可以帮助我们随机选取训练集和测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_full,Y,test_size=0.3,random_state=0)

# 让我们从最简单的形式开始，我们从这13个特征里只选择一个建立一个线性模型。
# 我们选择最后一个特征LSTAT （下层经济阶层百分比）
# 使用reshape将X转为为二维数组
X = X_train[:, 12].reshape((-1,1))  #X：LSTAT

#数据可视化
plt.scatter(X,Y_train,color='black')
plt.ylabel('Median Home Value')
plt.xlabel('LSTAT')
#plt.show() #看到什么规律了吗？

regressor = LinearRegression(normalize=True)#使用sklearn里线性回归的模块
regressor.fit(X,Y_train)
plt.scatter(X, Y_train, color='black')
plt.plot(X,regressor.predict(X),color='blue',linewidth=3)
plt.ylabel('Median Home Value')
plt.xlabel('LSTAT')
plt.show() # 使用抛物线拟合会不会更好？

# 我们可以查看beta1的估计值
# 我们可以这样理解beta1：当下层经济百分比增加1%，房屋平均中间价位降低$970
print('Coefficients:\n',regressor.coef_)#beta_1的点估计值
# 使用RMSE来检查预测误差，RMSE越大误差则越大。
X_1d_test = X_test[:, 12].reshape(-1, 1)
print(metrics.mean_squared_error(Y_test, regressor.predict(X_1d_test)))