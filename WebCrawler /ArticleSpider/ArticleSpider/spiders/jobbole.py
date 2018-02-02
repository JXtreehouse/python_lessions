# -*- coding: utf-8 -*-
import re
import datetime
from scrapy.http import Request
from urllib import parse

import scrapy
from ArticleSpider.items import JobBoleArticleItem
from ArticleSpider.utils.common import get_md5

class JobboleSpider(scrapy.Spider):
    name = 'jobbole'
    allowed_domains = ['blog.jobbole.com']
    start_urls = ['http://blog.jobbole.com/all-posts/']

    def parse(self, response):
        """
        1. 获取文章列表页的具体url，并交给scrapy下载 然后给解析函数进行具体字段的解析
        2. 获取下一页的url并交给scarpy进行下载, 下载完成后交给parse函数
        """

        #解析列表页中的所有url 并交给scrapy下载后进行解析
        post_nodes = response.css("#archive .floated-thumb .post-thumb a")
        for post_node in post_nodes:
            # 获取封面图url
            image_url = post_node.css("img::attr(src)").extract_first("")
            post_url = post_node.css("::attr(href)").extract_first("")
            url = parse.urljoin(response.url, post_url)
            # post_url 是我们每一页的具体的文章url。
            # 下面这个request是文章详
            # 情页面. 使用回调函数每下载完一篇就callback进行这一篇的具体解析。
            # 我们现在获取到的是完整的地址可以直接进行调用。如果不是完整地址: 根据response.url + post_url
            # def urljoin(base, url)完成url的拼接
            request = Request(url,meta={"front_image_url": image_url}, callback= self.parse_detail)
            yield request

        #提取下一页并交给scrapy进行下载
        next_url = response.css(".next.page-numbers::attr(href)").extract_first()
        if next_url:
            yield Request(url=parse.urljoin(response.url, next_url), callback=self.parse)

    def parse_detail(self, response):
        # 实例化item
        article_item = JobBoleArticleItem()

        print("通过item loader 加载item")
        # 通过item loader 加载item
        front_image_url = response.meta.get("front_image_url","") #文章封面图


        #提取文章的具体字段((xpath方式实现))
        title = response.xpath("//div[@class='entry-header']/h1/text()").extract_first("")
        create_date = response.xpath("//p[@class='entry-meta-hide-on-mobile']/text()").extract()[0].strip().replace(".","").strip()
        praise_nums = response.xpath("//span[contains(@class,'vote-post-up')]/h10/text()").extract()[0]

        fav_nums = response.xpath("//span[contains(@class,'bookmark-btn')]/text()").extract()[0]
        match_re = re.match(".*(\d+).*", fav_nums)
        if match_re:
            fav_nums = int(match_re.group(1))
        else:
            fav_nums = 0
        comment_nums = response.xpath("//a[@href='#article-comment']/span/text()").extract()[0]
        match_re = re.match(".*(\d+).*", comment_nums)
        if match_re:
            comment_nums = int(match_re.group(1))
        else:
            comment_nums = 0
        content = response.xpath("//div[@class='entry']").extract()[0]

        tag_list= response.xpath("//p[@class='entry-meta-hide-on-mobile']/a/text()").extract()
        # 去掉以评论结尾的字段
        tag_list = [element for element in tag_list if not element.strip().endswith("评论")]
        tags = ",".join(tag_list)

        # 为实例化后的对象填充值
        article_item["url_object_id"] = get_md5(response.url)
        article_item["title"] = title
        article_item["url"] = response.url
        try:
            create_date = datetime.datetime.striptime(create_date, "%Y/%m/%d").date()
        except Exception as e:
            create_date = datetime.datetime.now().date()
        article_item["create_date"] = create_date
        article_item["front_image_url"] = [front_image_url]
        article_item["praise_nums"] = praise_nums
        article_item["comment_nums"] = comment_nums
        article_item["fav_nums"] = fav_nums
        article_item["tags"] = tags
        article_item["content"] = content
        #print(tags)#职场,面试

        ## 已经填充好了值调用yield传输至pipeline
        yield article_item