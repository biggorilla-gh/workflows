"""
Authored by Wang-Chiew Tan
"""

# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy import signals
from scrapy.exporters import JsonLinesItemExporter


#
# this pipeline writes each item to the file specified. it
# gets called with each item.
#
class RestaurantSpiderPipeline(object):
    filename = ""

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        # get the specified filename to write to
        filename = settings.get("OUTFILE")
        pipeline = cls(filename)
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def __init__(self, filename):
        # open the file for writing
        self.file = open(filename, 'w+b')
    
    def spider_opened(self, spider):
        self.exporter = JsonLinesItemExporter(self.file)
        self.exporter.start_exporting()

    def spider_closed(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item

