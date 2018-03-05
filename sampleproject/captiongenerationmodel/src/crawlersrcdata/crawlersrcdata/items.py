# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class FlickrItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # img_url img_desc and img_id
    img_urls = scrapy.Field()
    img_descs = scrapy.Field()
    img_ids = scrapy.Field()
