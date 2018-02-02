# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import codecs
import json
import pymysql

from scrapy.pipelines.images import ImagesPipeline
from scrapy.exporters import JsonItemExporter

class ArticlespiderPipeline(object):
    def process_item(self, item, spider):
        return item

class MysqlPipeline(object):
    def __init__(self):
        # 获取一个数据库连接，注意如果是UTF-8类型的，需要制定数据库
        self.conn = pymysql.connect('127.0.0.1', 'root', 'wyc2016','article_spider', charset='utf8',use_unicode=True)
        self.cursor = self.conn.cursor()#获取一个游标

    def process_item(self, item, spider):
        insert_sql = """INSERT INTO jobboleArticle(title, url, create_date, fav_nums) VALUES(%s, %s, %s, %s )"""
        try:
            self.cursor.execute(insert_sql, (item["title"], item["url"], item["create_date"], item["fav_nums"]))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
        finally:
            self.conn.close()

class JsonWithEncodingPipeline(object):
    #自定义json文件的到出
    def __init__(self):
        self.file = codecs.open('article.json', 'w', encoding="utf-8")
    def process_item(self, item, spider):
        lines = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(lines)
        return item

    def spider_closed(self, spider):
        self.file.close()

class JsonExporterPipeline(object):
    #调用scrapy提供的JsonItemExporter  到出json文件
    def __init__(self):
        self.file = open('articleecport.json', 'wb')
        self.exporter = JsonItemExporter(self.file, encoding="utf-8", ensure_ascii=False)
        self.exporter.start_exporting()
    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()
    def process_item(self, item, spider):
        self.exporter.export_item(item=item)
        return item
#图片处理pipline
class ArticleImagePipeline(ImagesPipeline):
    def item_completed(self, results, item, info):
        for ok, value in results:
            image_file_path_ = value["path"]
        item["front_image_path"] = image_file_path_
        return item
