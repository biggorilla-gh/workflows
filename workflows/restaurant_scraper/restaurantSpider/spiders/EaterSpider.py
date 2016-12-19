"""
Authored by Wang-Chiew Tan
"""
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from restaurantSpider.items import RestaurantSpiderItem
from scrapy.crawler import CrawlerProcess
import scrapy
#import sys, os

#
# spider for crawling www.eater.com/review
#
class EaterSpider(scrapy.Spider):
    name = "EaterSpider"
    allowed_domains = []
    start_urls = []

    # url to page that contains reviews of highly rated restaurants
    urlstr = "http://www.eater.com/reviews/rating/"
    # we are scraping pages 1 to 25 of this url
    for i in xrange(1,25):
        start_urls.append(urlstr+str(i))

    print("=== Start URLs: {}".format(start_urls))

    def parse(self, response):
        print "=== Starting to crawl Eater.com reviews === "
        urls = response.selector.xpath('//h3/a[@data-analytics-link="review"]/@href').extract()
        titles = response.selector.xpath('//h3/a[@data-analytics-link="review"]/text()').extract()
        dates = response.selector.xpath('//div[@class="m-entry-box__body"]/p/span[@class="p-byline__time"]/text()').extract()

        items = []
        for j in xrange(0,len(urls)):
            # item(url,title,date,content) is defined in items.py
            i = RestaurantSpiderItem(url=urls[j], title=titles[j], date=dates[j])
            items.append(i)
            # start scraping the content
            request = scrapy.Request(url=urls[j], callback=self.parse_cafe, errback=self.parse_error)
            request.meta['item'] = i  # pass item information to pass to parse_cafe
            yield request

    # capture and print error messages on console if needed
    def parse_error(self, response):
        item = response.meta['item']
        print("=== Error on {} ===".format(item['url']))
        yield item

    def parse_cafe(self, response):
        item = response.meta['item']
        print("=== Retrieving {} ===".format(item['url']))
        # extracting all paragraphs from the article
        item['content'] = response.selector.xpath('//p').extract()
        yield item
