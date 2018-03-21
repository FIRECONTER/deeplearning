# -*- coding:utf-8 -*-
"""
Description: tokenizer in keras
    1) 简单的tokenizer 可以实现单词列表转换为单词与其序列的映射关系
    2) 这是一种非常简单的将单词转换为数字的方式 向量化文本的一种简单方式 返回文本中词在词典中的下标(从1开始)构成的序列
Author: allocator
"""

from keras.preprocessing.text import Tokenizer


if __name__ == '__main__':
    tokenizer = Tokenizer()
    # 常用方法
    # fit 方法用于训练模型 其中给的参数texts 是所有待训练的文本构成的list
    # 这个地方可以接受一个string 这样的话string 中的每一个字母会被作为文本进行fit 与理想不符合
    texts = ['hello this is the first text', 'this is the second text']
    tokenizer.fit_on_texts(texts)

    # 然后tokenizer 可以使用常见的属性 word_counts word_index word_docs 以及document_count
    # word_counts 返回一个有序dict key是word 而 value 是该单词在所有文本中出现的次数
    # word_index 返回的是一个普通dict key 是 word 而 index 是词在词典中的下标 从1开始计算
    # word_docs 返回时一个普通dict key 是 word 而 value 是出现该单词的文本数量
    print('current word_counts')
    print(tokenizer.word_counts)
    print('current word_docs')
    print(tokenizer.word_docs)
    print('current word_index')
    print(tokenizer.word_index)
    # 常用函数
    test = ['hello nice', 'it is very good']
    # texts_to_sequences 将一个 text 的list 转换为一个序列 序列中每一个元素对应于改text 的词在词典中的index值
    # 所以这个返回值一定是一个二维的序列. 每一个元素对应文本转换后的序列.
    # 同理如果给定一个字符串 那么字符串中每个字母会被当做文本进行转换 与理想结果不符合 所以一定传递text list型参数

    res = tokenizer.texts_to_sequences(test)
    print(res)
    # 返回结果[[5], [2]] 因为只有hello 和 is 在字典里面 而他们相应的index 为 5 和 2 所以作为返回结果返回.

    # 计算vocabulary len(tokenizer.word_index) + 1
