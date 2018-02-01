# https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143193331387014ccd1040c814dee8b2164bb4f064cff000

import re
print(re.match(r'^\d{3}\-\d{3,8}$','010-12345'))

print(re.match(r'^\d{3}\-\d{3,8}$','010 12345'))


## 切分字符串
print('a b   c'.split(' '))

print(re.split(r'\s+','a b  c'))


print(re.split(r'[\s\,]+','a,b,c d'))
#注意和下面这种写法的区别
print(re.split(r'[\s\,]+]','a,b,c d'))


print(re.split(r'[\s\,\;]+','a,b;;c  d'))

##分组

m = re.match(r'^(\d{3})-(\d{3,8})$','010-12345')
print(m)

print(m.group(0))
print(m.group(1))
print(m.group(2))

