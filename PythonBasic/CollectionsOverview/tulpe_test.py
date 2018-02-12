
# tulple 是个可迭代对象
name_tuple = ("alex1","alex2")

# for name in name_list:
#     print(name)
#

# tuple元素不可变 所以此处不能执行
# name_tuple[0] = "alex3"


#这里就可执行　因为在ｐｙｔｈｏｎ这样的动态语言中变量只是符号而已　不是内存区块
#name_tuple = ("alex2","alex3")


# 拆包
user_tuple = ("alex",27,173)
name, age, height = user_tuple
print(name,age,height)

user_tuple2 = ("alex",27,173,"NanJing","edu")
name2, *other = user_tuple2
print(name2, other)


# tulple 元素不可变  不是绝对的  比如：
name_tuple2 = ("alex",[27,173])
print(name_tuple2)
name_tuple2[1].append(22)
print(name_tuple2)

#tuple 可哈希 可作为dict的key　　而ｌｉｓｔ不可以
# 如果用c语言来类比  tuple对应的是struct 而list对应的是array
user_info_dict= {}

user_info_dict[user_tuple] = "alex"

pass


