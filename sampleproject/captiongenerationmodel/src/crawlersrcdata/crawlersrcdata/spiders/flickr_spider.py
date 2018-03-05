# -*- coding:utf-8 -*-
"""
Description: the spider of the flickr src data
"""


import scrapy

from crawlersrcdata.items import FlickrItem

class FlickrSpider(scrapy.spiders.Spider):
    name = 'flickr'
    allowed_domains = ['illinois.edu']
    start_urls = ['http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html']

    def parse(self, response):
        """Parse the response data from spider."""
        # print(response)
        # print(' type of current response is %s ' % type(response))
        # get all the picture links in the page
        # some links do not exist Image Not Found
        # one response with one spider item object
        img_items = response.xpath('//table/tr/td')
        img_urls = []
        img_descs = []
        img_ids = []
        invalid_image_num = 0
        valid_image_num = 0
        print(' image url and desc length is %d ' % len(img_items))
        n = int(len(img_items)/2)
        res_item = FlickrItem()
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
                img_ids.append('img'+str(valid_image_num))
                img_urls.append(img_url_item.xpath('a/@href').extract()[0])
                # get all the descriptions about every image
                current_txts = img_desc_item.xpath('ul/li/text()').extract()
                tmp_arr = []
                for txt_item in current_txts:
                    tmp_arr.append(txt_item.replace('\n', ''))
                img_descs.append(tmp_arr)
        print(' all the image items is %d ' % n)
        print(' the valid image items is %d ' % valid_image_num)
        print(' the invalid image items is %d ' % invalid_image_num)
        res_item['img_urls'] = img_urls
        res_item['img_descs'] = img_descs
        res_item['img_ids'] = img_ids
        return res_item