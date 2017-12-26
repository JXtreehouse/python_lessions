# coding: utf-8

"""
直接执行本文本，会在当前目前目录创建文件夹spider_res来保存结果
"""

import os
import time
import urllib2
import urlparse
from bs4 import BeautifulSoup  # 用于解析网页中文, 安装： pip install beautifulsoup4


def download(url, retry=2):
    """
    下载页面的函数，会下载完整的页面信息
    :param url:
    :param retry:
    :return:
    """
    print ("downloading: ", url)
    # 设置header信息，模拟浏览器请求
    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36'
    }
    try:
        request = urllib2.Request(url, headers=header)
        html = urllib2.urlopen(request).read()
    except urllib2.URLError as e:
        print ("download error: ", e.reason)
        html = None
        if retry > 0:
            if hasattr(e, 'code') and 500 <= e.code < 600:
                print (e.code)
                return download(url, retry - 1)
    time.sleep(1)
    return html


url_root = 'http://www.jianshu.com/'  # 网站根目录
url_seed = 'http://www.jianshu.com/c/9b4685b6357c?page=%d'  # 要爬取的页面地址模板
crawled_url = set()  # 已经爬取过的链接
flag = True

# step1 抓取文章链接
i = 1
while flag:
    url = url_seed % i
    i += 1

    html = download(url)
    if html == None:
        break

    soap = BeautifulSoup(html, "html.parser")
    links = soap.find_all('a', {'class': 'title'})
    if links.__len__() == 0:
        flag = False

    for link in links:
        link = link.get('href')
        if link not in crawled_url:
            realUrl = urlparse.urljoin(url_root, link)
            crawled_url.add(realUrl)  # 已爬取的页面记录下来，避免重复爬取
        else:
            print ('end')
            flag = False  # 结束抓取

paper_num = crawled_url.__len__()
print ('total paper num: ', paper_num)

# step2 抓取文章内容,并按标题和内容保存起来

for link in crawled_url:
    html = download(link)
    soap = BeautifulSoup(html, "html.parser")
    title = soap.find('h1', {'class': 'title'}).text
    content = soap.find('div', {'class': 'show-content'}).text

    # step3 保存爬取结果
    if os.path.exists('spider_res/') == False:
        os.mkdir('spider_res')

    file_name = 'spider_res/' + title + '.txt'
    if os.path.exists(file_name):
        # os.remove(file_name) # 删除文件
        continue  # 已存在的文件不再写
    file = open('spider_res/' + title + '.txt', 'wb')
    content = unicode(content).encode('utf-8', errors='ignore')
    file.write(content)
    file.close()