from bs4 import BeautifulSoup
import requests
headers = {
    'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 Mobile/14E5239e Safari/602.1'
}
url = 'https://www.tripadvisor.cn/Attractions-g294220-Activities-Nanjing_Jiangsu.html'
mb_data = requests.get(url,headers=headers)
soup = BeautifulSoup(mb_data.text,'html.parser')
imgs = soup.select('div.thumb.thumbLLR.soThumb > img')
for img in imgs:
    print(img.get('src'))