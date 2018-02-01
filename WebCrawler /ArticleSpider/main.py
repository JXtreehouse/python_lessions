__author__ = 'AlexZ33'
__date__ = '2018/1/30'

from scrapy.cmdline import execute
import sys
import os

# 将项目根目录加入系统环境变量中。
# os.path.abspath(__file__)为当前文件所在绝对路径
# os.path.dirname() 获取文件的父目录。

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print(os.path.abspath(__file__))
# /home/wyc/study/python_lessions/WebCrawler /ArticleSpider/main.py

print(os.path.dirname(os.path.abspath(__file__)))
#/home/wyc/study/python_lessions/WebCrawler /ArticleSpider


# 调用execute函数执行scrapy命令，相当于在控制台cmd输入该命令
# 那么就不用每次都运行上面scrapy crawl XXX（爬虫名字)，直接运行main.py就能启动爬虫了
# 可以传递一个数组参数进来:
if __name__ == '__main__':
    execute(["scrapy", "crawl", "jobbole",'--nolog'])
