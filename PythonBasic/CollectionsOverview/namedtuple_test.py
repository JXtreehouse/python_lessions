# reference:https://www.imooc.com/video/16363


# class User:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
# user = User(name = 'alex', age =27)
#
# print(user.name, user.age)

from collections import namedtuple

# 用namedtuple创建类 它是个类 不是一个对象

User = namedtuple("User", ["name", "age", "height"])
user = User(name="alex",age=27,height=1.73)
print(user.name,user.age,user.height)


# 用tulple初始化 namedtuple
User = namedtuple("User", ["name", "age", "height", "edu"])

user_tuple = ("alex", 27,173 ,"master")
# user = User(*user_tuple, "master")# 同　user = User("alex",27,173,"master")
# print(user.age, user.name, user.edu)


user_dict = {
    "name": "alex",
    "age": 27,
    "height": 173
}
#user2= User(**user_dict,edu = "phd")
#print(user2.age, user2.name, user2.edu)

# user3 = User._make(user_tuple)
print(user3.name)

# 函数参数
# def ask(name="alex"):
#     pass
def ask(*args, **kwargs):
    pass

#ask("alex",27)
ask(name="alex", age=27)