# reference: https://www.imooc.com/video/16367
from collections import OrderedDict

user_dict = OrderedDict()
user_dict["b"] = "alex2"
user_dict["a"] = "alex1"
user_dict["c"] = "alex3"


#print(user_dict.pop("a"))
print(user_dict.move_to_end("b"))
print(user_dict)

