#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import pandas as pd
import os
# df = pd.read_csv(u'data/dxy.csv', encoding='utf_8')
df = pd.read_excel(u'data/dxy.xlsx', encoding='utf_8')

#删除有缺失内容的行
df = df.dropna()

# df[u'相关疾病'][df[u'相关疾病'].isnull()]= ''

dfg = df.groupby(u'发帖用户id')
# 将发帖ｉｄ只出现一次的去除

oneceid = df.drop_duplicates([u'发帖用户id'],keep = False)[u'发帖用户id']


for idx in oneceid:

    df = df[df[u'发帖用户id']!= idx]
# 按照发帖用户ｉｄ对表排序
df = df.sort_values(by= [u"发帖用户id"],ascending=True)
# print(df.describe())
# df.to_excel('dxy2.xlsx')



