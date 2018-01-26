# -*-coding:utf-8 -*-
import requests
import csv

def get_positon(x):
    url = 'http://restapi.amap.com/v3/geocode/geo?'
    param = {'key':'c12648e53b8bb0322a83ff2c48ce27bb',
          'city':'南京',
          'address':x,
          'oupput':'JSON'
            }
    r = requests.post(url, data=param)
    s = r.json()
    geocode = s.setdefault('geocodes')
    if geocode:
        dist = geocode[0].setdefault('district')
        return dist
path = 'C:/Users/Administrator/Desktop/nj/noise_re.csv'
out = 'C:/Users/Administrator/Desktop/nj/noise_re_add_dist.csv'
with open(path, 'r') as csvfile:
    with open(out, 'w') as sfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(sfile)
        for row in reader:
            row_s = []
            dist = get_positon(row[4])
            for i in range(8):
                row_s.append(row[i])
            row_s.append(dist)
            writer.writerow(row_s)

