# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.contrib.pipeline.images import ImagesPipeline
from scrapy.exceptions import DropItem
import json

img_descs_filename = 'data/txt/image_descs.json'
img_urls_dir = 'data/img'
# deal with the image url and save the image file
class MyImagePipeline(ImagesPipeline):
    def get_media_requests(self, item, info):
        print(' get current media request ')
        for image_url in item['img_urls']:
            yield scrapy.Request(image_url)

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            raise DropItem("Item contains no images")
        item['image_paths'] = image_paths
        return item


# deal with the descs about the image
class MyDescPipeline(object):
    def process_item(self, item, spider):
        print(' desc pipeline works ')
        # write the description data to json file
        img_ids = item['img_ids']
        img_descs = item['img_descs']
        print(' img id size is %d and img desc size is %d ' % (len(img_ids), len(img_descs)))
        # save the text data to a dict and pre clean the text data
        res = dict()
        for i, item in enumerate(img_ids):
            res[item] = img_descs[i]
        json.dump(res, open(img_descs_filename, 'w'))
        print(' txt save task finished ')
        return item
