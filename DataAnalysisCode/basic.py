# 数据结构、函数、条件和循环
# 列表
l = ['l','i','s','t']
print(type(l))
# 元组中圆括号可以省略，但括号有助于快速识别元组
t = ('t','u','p','l','e')
print(type(t))

# tuple()函数可以把其他数据类型转化为元组
print(tuple(l))
print(type(tuple(l)))


# 索引和切片等列表操作也可以适用到元组上
print(t[2:4])

# 元组里的元素是不可修改的
# 列表中的元素是可修改的
l[0] = 'L'
print(l[0])


# 元组的元素不可修改，否则产生错误提示
#t[0] = 'T'
# print(t[0])


# 字典（dictionary）
# 在一对一的查找方面，字典比列表更方便
# countries and captials 是两个列表，记录了欧洲的一些国家及其对应的首都
countries = ['spain', 'france', 'germany', 'italy'] #国家
capitals = ['madrid', 'paris', 'berlin', 'rome'] #首都

# 打印德国首都
ind_ger = countries.index("germany")#先获得德国对应的数字索引
print(capitals[ind_ger])

# 用字典可以更好的实现这个功能
# 定义字典europe
europe = {'spain':'madrid', 'france':'paris','germany':'berlin','italy':'rome'}
print(europe['germany'])

#获取字典的keys和values
# 用keys()方法获取字典的索引
print(europe.keys())
# 用values()方法获取字典的值
print(europe.values())
# 用items()方法获取 key-value 键值对
print(europe.items())  # 每个元素是元组


#通过键 key 获取 value 值
# 输出france对应的值
print(europe['france'])
#我们也可以使用 get() 方法来获取key对应的值
print(europe.get('france'))

# 检查某个索引是否在字典里
print('britain' in europe)

# 当key不在字典中时，使用 [ ] 将出错
# europe['britain']
# 如果key不在字典里，get方法默认返回None
print(europe.get('britain'))


# 也可以自定义返回值，如果key不在字典里,返回‘Unknown’
print(europe.get('britain', 'Unknown'))

# 字典的增删改

#增加新的键-值对
europe['britain'] = 'london'
# 注意这里的输出顺序和我们输入的顺序不一致，也再一次说明了字典是无序的
print(europe)

#字典的值是可修改的
europe['britain']  = 'London'
print(europe)