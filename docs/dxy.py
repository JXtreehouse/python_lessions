#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
# df = pd.read_csv(u'data/dxy.csv', encoding='utf_8')
df = pd.read_excel(u'data/dxy.xlsx', encoding='utf_8')
df = df.dropna()

# df[u'相关疾病'][df[u'相关疾病'].isnull()]= ''
dfg = df.groupby(u'发帖用户id')

oneceid = df.drop_duplicates([u'发帖用户id'],keep = False)[u'发帖用户id']
for idx in oneceid:
    df = df[df[u'发帖用户id']!= idx]
df = df.sort_values(by= [u"发帖用户id"],ascending=True)
df.to_excel('dxy2.xlsx')


