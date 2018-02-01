#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# 试写一个验证Email地址的正则表达式。版本一应该可以验证出类似的Email：
#someone@gmail.com
#bill.gates@microsoft.com
###

import re
addr = 'someone@gmail.com'
addr2 = 'bill.gates@microsoft.com'
def is_valid_email(addr):
    if re.match(r'[a-zA-Z_\.]*@[a-aA-Z.]*',addr):
        return True
    else:
        return False

print(is_valid_email(addr))
print(is_valid_email(addr2))

# 版本二可以提取出带名字的Email地址：
# <Tom Paris> tom@voyager.org => Tom Paris
# bob@example.com => bob

addr3 = '<Tom Paris> tom@voyager.org'
addr4 = 'bob@example.com'

def name_of_email(addr):
    r=re.compile(r'^(<?)([\w\s]*)(>?)([\w\s]*)@([\w.]*)$')
    if not r.match(addr):
        return None
    else:
        m = r.match(addr)
        return m.group(2)

print(name_of_email(addr3))
print(name_of_email(addr4))