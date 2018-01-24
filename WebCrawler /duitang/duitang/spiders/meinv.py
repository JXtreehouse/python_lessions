# -*- coding: utf-8 -*-
import scrapy
from duitang.items import DuitangItem

class MeiNvSpider(scrapy.Spider):
    name = 'meinv'
    allowed_domains = ["duitang.com"]
    page = 1
    start_urls = []
    for page in range(1, 100):
        start_urls.append("https://www.duitang.com/search/?page=%s&kw=keyword&type=feed" % page)

    def parse(self, response):
        for sel in response.xpath('//a[@class="a"]/img/@src'):
            item = DuitangItem()
            detaillink = sel.extract().replace('.thumb.224_0', '')
            item['image_urls'] = [detaillink]
            yield item




#http://www.duitang.com/search/?page=%s&kw=%E7%BE%8E%E5%A5%B3&type=feed