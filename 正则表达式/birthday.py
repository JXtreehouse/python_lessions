import re

line = "XXX出生于2011年5月30日"
# line = "XXX出生于2011/5/30"
# line = "XXX出生于2011-5-30"
# line = "XXX出生于2011-05-30"
# line = "XXX出生于2011-05"

regex_str = ".*出生于(\d{4}[年/-]\d{1,2}([月/-]\d{1,2}|$))"
match_obj = re.match(regex_str, line)
if match_obj:
    print(match_obj.group(1))