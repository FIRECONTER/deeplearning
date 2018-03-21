# -*- coding:utf-8 -*-
"""
Description: keras 中的preprocessing package的一些使用
Author: allocator
"""
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


if __name__ == '__main__':
    # pad_sequences and to_categorical
    tokenizer = Tokenizer()
    texts = ['this is the first test doc', 'this is the second test doc']
    tokenizer.fit_on_texts(texts)
    # index of 0 is reversed
    vocab_size = len(tokenizer.word_index) + 1
    # calculate the vocab_size
    # pad_sequences 负责填充或者截断list(根据maxlen 参数确定是填充还是截断) 并且返回填充后的ndarray 类型序列
    # 填充和截断可以选择是前端还是尾部 传递的参数必须是 可迭代对象构成的list 所以一般是二维数组
    # pad_sequences(sequences, maxlen, padding, truncating)

    # to_categorical 方法
    # 完成类别向量映射到二值矩阵
    # 目的是在一些学习模型里面需要使用categorical_crossentropy 多类对数损失函数
    # 使用以上的目标函数来train 模型的时候需要将标签向量(nb_samples) 转换为(nb_samples, nb_classes)
    # 的二值序列作为y值 而keras.utils 中的to_categorical(y, nb_classes) 目的就是完成这样的一个标签向量向二值函数的转换过程
    # test
    y = [1, 2, 3, 4, 5]
    res = to_categorical(y)
    print(res)
    print(type(res))
    print(res.shape)
