#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
# df = pd.read_csv(u'data/dxy.csv', encoding='utf_8')
df = pd.read_excel(u'data/dxy-selected.xlsx', encoding='utf_8')
df.head()
dfg_count1 = df.groupby(u'发帖用户id').count()

dfg2=df.groupby([u'发帖用户id',u'科室'])
dfg_count2 = dfg2.count()
dfg_count3 = df.groupby([u'发帖用户id',u'相关疾病']).count()

# for i in dfg_count3:
#     print(i)


print(dfg_count3)
#dfg_count2.to_excel('dxy-selected2.xlsx')



