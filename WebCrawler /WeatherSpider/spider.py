# python2.7

from bs4 import BeautifulSoup
import requests
import time
# pip3 install echarts-python
from echarts import Echart, Legend, Bar, Axis

TEMPERATURE_LIST = []

CITY_LIST = []
MINTEMP_LIST = []

def get_temperature(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'http://www.weather.com.cn/textFC/hb.shtml',
        'Host': 'www.weather.com.cn'
    }

    req = requests.get(url, headers=headers)

    content = req.content

    soup = BeautifulSoup(content, 'lxml')

    conMidtab = soup.find('div', class_='conMidtab')
    conMidtab2_list = conMidtab.find_all('div', class_='conMidtab2')
    for x in conMidtab2_list:
        tr_list = x.find_all('tr')[2:]
        province = ''
        mintemp = 0
        for index, tr in enumerate(tr_list):
            if index == 0:
                td_list = tr.find_all('td')
                province = td_list[0].text.replace('\n', '')
                city = td_list[1].text.replace('\n', '')
                mintemp = td_list[7].text.replace('\n', '')
            else:
                td_list = tr.find_all('td')
                city = td_list[0].text.replace('\n', '')
                mintemp = td_list[6].text.replace('\n', '')
            #print('%s|%s' % (province + city, mintemp))
            TEMPERATURE_LIST.append({
                'city' : province + city,
                'mintemp': mintemp
            })
            CITY_LIST.append(province + city)
            MINTEMP_LIST.append(mintemp)

def main():
    urls = ['http://www.weather.com.cn/textFC/hb.shtml',
            'http://www.weather.com.cn/textFC/db.shtml',
            'http://www.weather.com.cn/textFC/hd.shtml',
            'http://www.weather.com.cn/textFC/hz.shtml',
            'http://www.weather.com.cn/textFC/hn.shtml',
            'http://www.weather.com.cn/textFC/xb.shtml',
            'http://www.weather.com.cn/textFC/xn.shtml']
    for url in urls:
        get_temperature(url)
        time.sleep(2)

    TOP20_MIN_LIST = MINTEMP_LIST[0:20]
    TOP20_CITY_LIST = CITY_LIST[0:20]
    print(TOP20_CITY_LIST)

    echart = Echart('全国最低温度统计', 'Author: AlexZ33')
    bar = Bar('最低温度', TOP20_MIN_LIST)
    axis = Axis('category', 'bottom', data = TOP20_CITY_LIST)

    echart.use(bar)
    echart.use(axis)
    echart.use(Legend(['℃']))
    echart.plot()




if __name__ == '__main__':
    main()