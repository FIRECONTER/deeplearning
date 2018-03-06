# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.contrib.pipeline.images import ImagesPipeline
from scrapy.exceptions import DropItem
import os

img_descs_folder = 'data/txt'
img_urls_dir = 'D:\\Alex\\Learn\\BigData\\deeplearning\\sampleproject\\captiongenerationmodel\\src\\crawlersrcdata\\data\\img\\'
data_log_dir = 'data/logs'
# deal with the image url and save the image file
# write the error image information to the log file


class MyImagePipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        if item['isvalid']:
            yield scrapy.Request('https:'+item['image_url'])

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            print(' download current image error with id %s ' % item['image_id'])
            with open(os.path.join(data_log_dir, 'datalog.txt'), 'a+') as datalog:
                datalog.write(' download error image id %s and results status %s and url is %s \n' % (item['image_id'], results[0][0], results[0][1]['url']))
            raise DropItem("Item contains no images")
        old_name = image_paths[0]
        os.rename(img_urls_dir+old_name,img_urls_dir+'full\\'+item['image_id']+'.jpg')
        return item

# deal with the descs about the image
class MyDescPipeline(object):
    def process_item(self, item, spider):
        # write the description data to json file
        # id is the filename and content is the desc
        if item['isvalid']:
            # save the text to a folder
            with open(os.path.join(img_descs_folder, item['image_id']+'.txt'), 'w') as file:
                file.writelines(item['image_desc'])
        return item
