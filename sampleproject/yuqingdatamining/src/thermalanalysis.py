#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：测试gensim使用，处理中文语料
日期：2018年1月3日
"""
from gensim.models import word2vec,Phrases
from gensim.models.doc2vec import Doc2Vec,TaggedDocument,TaggedLineDocument
import logging
import warnings
from pprint import  pprint
import multiprocessing
warnings.filterwarnings("ignore")
cores = multiprocessing.cpu_count()
# 训练主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences =word2vec.Text8Corpus('../data/jiayuqingrecent.txt')# 加载语料
model = word2vec.Word2Vec(sentences,sg=0, min_count=10,size=100,seed=1,workers=cores,window=5)# 训练skip-gram模型; 默认window=5
#在训练语料中，寻找与贾跃亭相关的关联词
model.wv.most_similar(['贾跃亭'],topn=15)
