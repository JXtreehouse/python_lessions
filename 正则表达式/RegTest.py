import re
line = 'jwxddxsw33'
if line == "jxdxsw33":
    print("yep")
else:
    print("no")

regex_str = "^j.*"
if re.match(regex_str, line):
    print("yes")

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