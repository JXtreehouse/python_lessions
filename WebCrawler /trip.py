from bs4 import BeautifulSoup
import requests

url = 'https://www.tripadvisor.cn/Attractions-g294220-Activities-Nanjing_Jiangsu.html'
urls = ['https://www.tripadvisor.cn/Attractions-g294220-Activities-oa{}-Nanjing_Jiangsu.html#FILTERED_LIST'.format(str(i)) for i in range(30,800,30)]

def get_attraction(url, data=None):
    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text, 'html.parser')
    # print(soup)
    # 使用BeautifulSoup对html解析时，当使用css选择器，对于子元素选择时，要将nth-child改写为nth-of-type才行
    #titles = soup.select('#taplc_attraction_coverpage_attraction_0 > div:nth-of-type(1) > div > div > div.shelf_item_container > div:nth-of-type(1) > div.poi > div > div.item.name > a')
    titles = soup.select('a.poiTitle')
    # imgs = soup.select('img.photo_image')
    imgs = soup.select('img[width="200"]')

    # 把信息转入字典
    for title, img in zip(titles,imgs):
        data = {
            'title': title.get_text(),
            'img': img.get('src'),
        }
        print(data)


for single_url in urls:
    get_attraction(single_url)
