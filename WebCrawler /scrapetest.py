from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen("http://jxdxsw.com/")
bsobj = BeautifulSoup(html.read())
print(bsobj.prettify())

print("-----------------------------我是分割线---------------------------")
print(bsobj.title)

print("-----------------------------我是分割线---------------------------")
print(bsobj.find_all('a'))
