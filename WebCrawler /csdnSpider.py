#本实例用于获取指定用户csdn的文章名称、连接、阅读数目
import urllib
import re
from bs4 import BeautifulSoup

#csdn不需要登陆，也不需要cookie,也不需要设置header

print('=======================csdn数据挖掘==========================')
urlstr="http://blog.csdn.net/luanpeng825485697?viewmode=contents"
host = "http://blog.csdn.net/luanpeng825485697"  #根目录

alllink = [urlstr] #所有需要遍历的网址
data = {}

def getdata(html, reg): #从字符串中安装正则表达式获取值
    pattern = re.compile(reg)
    items = re.findall(pattern, html)
    for item in items:
        urlpath = urllib.parse.urljoin(urlstr, item[0])  #将相对地址，转化为绝对地址
        print(urlpath)
        if not hasattr(object, urlpath):
            data[urlpath] = item
            print(urlpath,'  ',end=' ')#python3中end表示结尾符，这里不换行
            print(item[2], '  ', end= ' ')
            print(item[1])

# 根据一个网址获取相关连接并添加到集合中
def getlink(url, html):
    soup = BeautifulSoup(html, 'html5lib') #使用html5lib解析，所以需要提前安装好html5lib包
    for tag in soup.find_all('a'):  #从文档中找到所有<a>标签的内容
        link = tag.get('href')
        newurl = urllib.parse.urljoin(url, link) #在指定网址中的连接的绝对连接
        if host not in newurl:  # 如果是站外连接，则放弃
            continue
        if newurl in alllink:  #不添加已经存在的网址
            continue
        if not  "http://blog.csdn.net/luanpeng825485697/article/list"


