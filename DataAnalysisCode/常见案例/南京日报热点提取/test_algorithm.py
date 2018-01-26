# -*-coding: utf-8 -*-
from datetime import datetime
import csv
from collections import defaultdict
import jieba
from location_.gsdmm import *
import re
import numpy as np
import operator
import os
orders = []
stop_words = []
out = 'C:/Users/Administrator/Desktop/out'
with open('C:/Users/Administrator/Desktop/stop_words.csv','r') as stop_csv:
    readers = csv.reader(stop_csv)
    for r in readers:
        stop_words.append(r[0])
districts = set()
date = set()
with open('D:/njData/noven_noise.csv', 'r') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        # d = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        # day_string = d.strftime('%Y-%m-%d')
        day_string = row[1].split()[0]
        date.add(day_string)
        if row[4]:
            districts.add(row[4])
        orders.append([row[0], day_string, row[2], row[3], row[4]])
orders_per_day = defaultdict(list)
for d in date:
    orders_per_day[d] = [order for order in orders if order[1] == d]
for key, val in orders_per_day.items():
    district_ori = defaultdict(list)
    district_ord = defaultdict(list)
    district_dict = defaultdict(list)
    order_loc = []
    order_N = []
    for district in districts:
        for ord in val:
            str_or = re.sub('\W+', '', ord[2])
            remove_order = re.sub('[a-zA-Z0-9_]', "", str_or)
            segs = jieba.cut(remove_order)
            final = []
            for seg in segs:
                if seg not in stop_words:
                    final.append(seg)
            space_linked = ''.join(f for f in final)
            # aa = [address, address, address, space_linked]
            # order_address = ' '.join(oa for oa in aa)
            # # linked_order = ''.join(f for f in final)
            # order_N.append([ord[i] for i in range(5)])
            # order_loc.append(space_linked)
            if ord[4] == district:
                district_ori[district].append([ord[i] for i in range(5)])
                district_dict[district].append(ord[3])
                district_ord[district].append(space_linked)
    # order_Narr = np.array(order_N)
    # 分别对每个区聚类
    class_orders = defaultdict(list)
    class_orders_ori = defaultdict(list)
    final_class = defaultdict(list)
    for dis, locs in district_dict.items():
        mgp = MovieGroupProcess(K=100, alpha=0.1, beta=0.1, n_iters=200)
        labels = mgp.fit(docs=locs, vocab_size=30)
        num = np.unique(labels)
        dis_ord_array = np.array(district_ord[dis])
        dis_ori_array = np.array(district_ori[dis])
        for i in range(num.shape[0]):
            label = num[i]
            class_list = dis_ord_array[labels == label].tolist()
            class_list_ori = dis_ori_array[labels == label].tolist()
            new_key = ''.join([key, str(label), dis])
            class_orders[new_key] = class_list
            class_orders_ori[new_key] = class_list_ori
    # 分别对根据区分类的结果在对工单进行聚类
    for new_k, order_c in class_orders.items():
        class_ = np.array(class_orders_ori[new_k])
        mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=60)
        lbs = mgp.fit(docs=order_c, vocab_size=100)
        num_lb = np.unique(lbs)
        for j in range(num_lb.shape[0]):
            lb = num_lb[j]
            class_final = class_[lbs == lb].tolist()
            newest_key = ''.join([new_k, str(j), str(lb), new_k])
            final_class[newest_key] = class_final
    label_len = []
    # 写入csv
    with open(os.path.join(out, ''.join([key, '.csv'])), 'w') as Wcsv:
        writer = csv.writer(Wcsv)
        for key_ds, val in final_class.items():
            label_len.append([key_ds, len(val)])
        sort_value = sorted(label_len, key=operator.itemgetter(1), reverse=True)
        for sv in sort_value:
            key_sv = sv[0]
            for label_order in final_class[key_sv]:
                    label_order.append(key_sv)
                    writer.writerow(label_order)




