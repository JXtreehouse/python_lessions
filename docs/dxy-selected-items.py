#!/usr/bin/python2.7
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import os
# df = pd.read_csv(u'data/dxy.csv', encoding='utf_8')
df = pd.read_excel(u'data/dxy-selected-items.xlsx', encoding='utf_8')
# dfg_count1 = df.groupby(u'发帖用户id').count()
#
# dfg2=df.groupby([u'发帖用户id',u'科'])
# dfg_count2 = dfg2.count()
dfg_count3 = df.groupby([u'发帖用户id',u'科室']).count()
dfg_count4 = df.groupby([u'发帖用户id',u'所属栏目']).count()
# for i in dfg_count3:
#     print(i)

#
# dfg_count3.to_excel('dxy-selected-items2.xlsx')

print(dfg_count3.join(dfg_count4,how='outer'))



