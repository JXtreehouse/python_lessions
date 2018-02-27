# reference:https://www.imooc.com/video/16368

from collections import ChainMap
user_dict1 = {"a": "alex1", "b": "alex2"}
user_dict2 = {"c": "alex2", "d": "alex3"}

# 遍历数据
# 方法１
print("------------方法1----------------")
for key, value in user_dict1.items():
    print(key, value)
for key, value in user_dict2.items():
    print(key, value)

# 方法2

print("------------方法2----------------")
new_dict = ChainMap(user_dict1, user_dict2)

for key, value in new_dict.items():
    print(key, value)

#　
print(new_dict["c"])
print(new_dict.maps)
new_dict.maps[0]["a"] = "Tom"
print(new_dict.maps)
