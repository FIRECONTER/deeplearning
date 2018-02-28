#-*- coding:utf-8 -*-
"""
Pre handle the train data set
"""
import collections
import nltk
import numpy as np


if __name__ == '__main__':
    train_file_path = './srcdata/train.txt'
    MAX_LEN = 0
    word_freq = collections.Counter()
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, sentence = line.strip().split('\t')
            words = nltk.word_tokenize(sentence.lower())
            if len(words) > MAX_LEN:
                MAX_LEN = len(words)
            for word in words:
                word_freq[word] = word_freq[word] + 1
    print(' train set max sentence length is %s and the word freq is %s ' % (MAX_LEN, len(word_freq)))
    
