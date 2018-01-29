import re
line = 'jwxddxsw33'
if line == "jxdxsw33":
    print("yep")
else:
    print("no")

# ^ 限定以什么开头
regex_str = "^j.*"
if re.match(regex_str, line):
    print("yes")
#$限定以什么结尾
regex_str1 = "^j.*3$"
if re.match(regex_str, line):
    print("yes")

regex_str1 = "^j.3$"
if re.match(regex_str, line):
    print("yes")
# 贪婪匹配
regex_str2 = ".*(d.*w).*"
match_obj = re.match(regex_str2, line)
if match_obj:
    print(match_obj.group(1))
# 非贪婪匹配
# ？处表示遇到第一个d 就匹配
regex_str3 = ".*?(d.*w).*"
match_obj = re.match(regex_str3, line)
if match_obj:
    print(match_obj.group(1))
# * 表示>=0次　　＋　表示　>=0次
# ? 表示非贪婪模式
# + 的作用至少>出现一次  所以.+任意字符这个字符至少出现一次
line1 = 'jxxxxxxdxsssssswwwwjjjww123'
regex_str3 = ".*(w.+w).*"
match_obj = re.match(regex_str3, line1)
if match_obj:
    print(match_obj.group(1))
# {2}限定前面的字符出现次数 {2,}2次以上 {2,5}最小两次最多5次
line2 = 'jxxxxxxdxsssssswwaawwjjjww123'
regex_str3 = ".*(w.{3}w).*"
match_obj = re.match(regex_str3, line2)
if match_obj:
    print(match_obj.group(1))

line2 = 'jxxxxxxdxsssssswwaawwjjjww123'
regex_str3 = ".*(w.{2}w).*"
match_obj = re.match(regex_str3, line2)
if match_obj:
    print(match_obj.group(1))

line2 = 'jxxxxxxdxsssssswbwaawwjjjww123'
regex_str3 = ".*(w.{5,}w).*"
match_obj = re.match(regex_str3, line2)
if match_obj:
    print(match_obj.group(1))

# | 或

line3 = 'jx123'
regex_str4 = "((jx|jxjx)123)"
match_obj = re.match(regex_str4, line3)
if match_obj:
    print(match_obj.group(1))
    print(match_obj.group(2))
# [] 表示中括号内任意一个
line4 = 'ixdxsw123'
regex_str4 = "([hijk]xdxsw123)"
match_obj = re.match(regex_str4, line4)
if match_obj:
    print(match_obj.group(1))
# [0,9]{9} 0到9任意一个 出现9次（9位数）
line5 = '15955224326'
regex_str5 = "(1[234567][0-9]{9})"
match_obj = re.match(regex_str5, line5)
if match_obj:
    print(match_obj.group(1))
# [^1]{9}
line6 = '15955224326'
regex_str6 = "(1[234567][^1]{9})"
match_obj = re.match(regex_str6, line6)
if match_obj:
    print(match_obj.group(1))

# [.*]{9} 中括号中的.和*就代表.*本身
line7 = '1.*59224326'
regex_str7 = "(1[.*][^1]{9})"
match_obj = re.match(regex_str7, line7)
if match_obj:
    print(match_obj.group(1))

#\s 空格
line8 = '你 好'
regex_str8 = "(你\s好)"
match_obj = re.match(regex_str8, line8)
if match_obj:
    print(match_obj.group(1))

# \S 只要不是空格都可以（非空格）
line9 = '你真好'
regex_str9 = "(你\S好)"
match_obj = re.match(regex_str9, line9)
if match_obj:
    print(match_obj.group(1))

# \w  任意字符 和.不同的是 它表示[A-Za-z0-9_]
line9 = '你adsfs好'
regex_str9 = "(你\w\w\w\w\w好)"
match_obj = re.match(regex_str9, line9)
if match_obj:
    print(match_obj.group(1))

line10 = '你adsf_好'
regex_str10 = "(你\w\w\w\w\w好)"
match_obj = re.match(regex_str10, line10)
if match_obj:
    print(match_obj.group(1))
#\W大写的是非[A-Za-z0-9_]
line11 = '你 好'
regex_str11 = "(你\W好)"
match_obj = re.match(regex_str11, line11)
if match_obj:
    print(match_obj.group(1))

# unicode编码 [\u4E00-\u\9FA5] 表示汉字
line12= "镜心的小树屋"
regex_str12= "([\u4E00-\u9FA5]+)"
match_obj = re.match(regex_str12,line12)
if match_obj:
    print(match_obj.group(1))

print("-----贪婪匹配情况----")
line13 = 'reading in 镜心的小树屋'
regex_str13 = ".*([\u4E00-\u9FA5]+树屋)"
match_obj = re.match(regex_str13, line13)
if match_obj:
    print(match_obj.group(1))

print("----取消贪婪匹配情况----")
line13 = 'reading in 镜心的小树屋'
regex_str13 = ".*?([\u4E00-\u9FA5]+树屋)"
match_obj = re.match(regex_str13, line13)
if match_obj:
    print(match_obj.group(1))

#\d数字
line14 = 'XXX出生于2011年'
regex_str14 = ".*(\d{4})年"
match_obj = re.match(regex_str14, line14)
if match_obj:
    print(match_obj.group(1))

regex_str15 = ".*?(\d+)年"
match_obj = re.match(regex_str15, line14)
if match_obj:
    print(match_obj.group(1))