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
    # each item contains just one image and the related text information
    image_url = scrapy.Field()
    image_desc = scrapy.Field()
    image_id = scrapy.Field()
    image_path = scrapy.Field()
    isvalid = scrapy.Field()
