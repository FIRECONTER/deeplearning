# -*- coding:utf-8 -*-
"""
Description: the spider of the flickr src data
"""


import scrapy

from crawlersrcdata.items import FlickrItem

class FlickrSpider(scrapy.spiders.Spider):
    name = 'flickr'
    # very important allowed domains if the data come from different domain
    allowed_domains = ['illinois.edu', 'flickr.com']
    start_urls = ['http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html']

    def parse2(self, response):
        """Parse the url form the image site url"""
        item = response.meta['item']
        image_urls = response.xpath('//img/@src').extract()
        if len(image_urls) == 1:
            print(' current image does not exist because image length is 1')
            item['isvalid'] = False
        else:
            item['isvalid'] = True
            item['image_url'] = image_urls[0]
        return item

    def parse(self, response):
        """Parse the response data from spider."""
        # print(response)
        # print(' type of current response is %s ' % type(response))
        # get all the picture links in the page
        # some links do not exist Image Not Found
        # one response with one spider item object
        img_items = response.xpath('//table/tr/td')
        invalid_image_num = 0
        valid_image_num = 0
        print(' image url and desc length is %d ' % len(img_items))
        n = int(len(img_items)/2)
        for i in range(n):
            img_url_item = img_items[2*i]
            img_desc_item = img_items[2*i+1]
            if len(img_url_item.xpath('ul')) == 0 and len(img_url_item.xpath('a')) == 0:
                print(' Maybe image does not exist ')
                print(img_url_item.extract()[0])
                invalid_image_num += 1
            else:
                # get the url and image descriptions
                valid_image_num += 1
                item = FlickrItem()
                item['image_id'] = 'img_' + str(valid_image_num)
                image_site_url = img_url_item.xpath('a/@href').extract()[0]
                # get all the descriptions about every image
                current_txts = img_desc_item.xpath('ul/li/text()').extract()
                # tmp_arr = []
                # for txt_item in current_txts:
                #     tmp_arr.append(txt_item.replace('\n', ''))
                item['image_desc'] = current_txts
                yield scrapy.Request(image_site_url, callback=self.parse2, meta={'item': item})
        print(' all the image items is %d ' % n)
        print(' the valid image items is %d ' % valid_image_num)
        print(' the invalid image items is %d ' % invalid_image_num)