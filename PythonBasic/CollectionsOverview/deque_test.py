# reference: https://www.imooc.com/video/16365

# deque  双端队列

from collections import deque
import copy

# pop
user_list = ["alex1", "alex2"]
user_name =  user_list.pop()

print(user_name, user_list)

# 双端操作

#user_deque = deque(["alex1","alex2","alex3"])
# user_deque.appendleft("alex7")

## 浅拷贝 : 只拷贝里面的元素　如果是不可变元素直接复制一个　　如果是可变原始其实是指向其存储值　　比如ｌｉｓｔ
# user_deque2 = user_deque.copy()
# user_deque2[1] = "Tom"



user_deque = deque(["alex1",["alex2","alex4"],"alex3"])
user_deque2 = user_deque.copy()
user_deque2[1].append("Tom")

print(user_deque, user_deque2)
print(id(user_deque),id(user_deque2))
print(user_deque)


## 深拷贝 copy.deepcopy


# deque GIL是线程安全的　　list 不是线程安全的