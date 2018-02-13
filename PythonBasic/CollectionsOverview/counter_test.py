# reference: https://www.imooc.com/video/16366

from collections import Counter
# 统计list
users = ["alex1","alex2","alex1","alex2","alex3"]
user_counter = Counter(users)
print(user_counter)

#统计字符串
user_counter = Counter("abbafafpskaag")


print(user_counter.most_common(3))
user_counter.update("dfsd")
print(user_counter)
print(user_counter['a'])
print(sorted(user_counter))
print(user_counter.elements())
print(sorted(user_counter.elements()))
print(''.join(sorted(user_counter.elements())))
#