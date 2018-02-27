from bs4 import BeautifulSoup
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36',
    'Upgrade-Insecure-Requests': '1',
    'Referer':'http://www.weather.com.cn/textFC/hb.shtml',
    'Host':'www.weather.com.cn'
}

req = requests.get('http://www.weather.com.cn/textFC/hb.shtml', headers=headers)

content = req.content

soup = BeautifulSoup(content, 'lxml')

conMidtab = soup.find('div', class_='conMidtab')
conMidtab2_list = conMidtab.find_all('div',class_='conMidtab2')
for x in conMidtab2_list:
    tr_list = x.find_all('tr')
    province_tr = tr_list[2]
    td_list = province_tr.find_all('td')
    province_td = td_list[0]
    province = province_td.text
    print(province.replace('\n',''))
