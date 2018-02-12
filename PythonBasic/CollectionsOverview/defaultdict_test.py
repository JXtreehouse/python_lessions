# reference: https://www.imooc.com/video/16364

from collections import defaultdict

# # 统计元素出现次数
# # 方法１
#
# print("--------------方法１------------------")
# user_dict = {}
# users = ["alex1","alex2","alex3","alex2","alex1"]
# for user in users:
#     print(user)
#     if user not in user_dict:
#         user_dict[user] = 1
#     else:
#         print(user_dict[user])
#         user_dict[user] += 1
# print(user_dict)
#
# # 方法２
#
# print("--------------方法２------------------")
# user_dict = {}
# users = ["alex1","alex2","alex3","alex2","alex1"]
# for user in users:
#     user_dict.setdefault(user, 0)
#     user_dict[user] += 1
#
# print(user_dict)

# 方法３　
print("--------------方法3------------------")
default_dict = defaultdict(int)

users = ["alex1","alex2","alex3","alex2","alex1"]
for user in users:
    default_dict[user] += 1
pass



# defaultdict只能传递一些可调用的对象　list int

def gen_default():
    return {
        "name": "",
        "nums": 0
    }
default_dict = defaultdict(gen_default)
default_dict["group1"]
pass

# group_dict = {
#     "group1": {
#         "name":"",
#         "nums": 0
#     }
# }
