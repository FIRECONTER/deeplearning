# -*- coding:utf-8 -*-
"""
Description: pooling layer in the dl model
    1) 这里的pooling 层其实是降采样层 目的是将输出的feature map 的维度降低 放置训练模型过拟合 同时进一步提取关键信息 降低
    输入数据的不稳定性对模型的影响.
    2) 了解一下keras 中的pooling 层类型有哪些 以及一些用途
    3) 学习使用sklearn 中的minist 数据集合
    4) 完成一个完整的简单的CNN分类神经网络 作用于 minist dataset 核心是对于pooling 和 convNet 的理解
Author: allocator
Time: 26/03/2017
"""
import numpy as np
import numpy.random as rd
from sklearn.datasets import fetch_mldata
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input
import matplotlib.pyplot as plt


def load_minist(name):
    """Print the minist dataset info."""
    # fetch_mldata 从mldata.org download data 并且返回一个类似于dict类型的数据
    # 下载失败可能需要翻墙 下载网站为mldata.org
    minist_dataset = fetch_mldata(name)
    # Bunch type 数据
    print(type(minist_dataset))
    # 一般情况下主要有四个key data target DESCR 以及COL_NAMES
    print(minist_dataset.keys())
    # data 为ndarray 类型
    print('current data label type %s' % type(minist_dataset['data']))
    # descr 为整个数据集合的描述 str类型
    print('current DESCR label type %s' % type(minist_dataset['DESCR']))
    print(minist_dataset['DESCR'])
    # target 类型同样为ndarray 类型
    print('current target label type %s' % type(minist_dataset['target']))
    # COL_NAME 类型为List 类型
    print('current COL_NAMES label type %s' % type(minist_dataset['COL_NAMES']))
    print(minist_dataset['COL_NAMES'])
    data = minist_dataset['data']
    target = minist_dataset['target']
    print('current data shape (%d,%d)' % data.shape)
    print('current target shape (%d)' % target.shape)
    # data 数据中存储的是minist 图像数据的灰度值 每一行为一个向量 数值类型为numpy.uint8 且大小为784 是 28*28 的图像的向量表达
    # target 是一个list 存储每一个data 对应的标签 0.0-9.0 类型为numpy.float64
    return minist_dataset


def print_minit_demo(data):
    """Visualization of the minist data."""
    data_size = len(data)
    COLS = 5
    rows = int(np.ceil(data_size/COLS))
    fig, ax = plt.subplots(
        ncols = COLS,
        nrows = rows,
        sharex = True,
        sharey = True)
    ax = ax.flatten()
    for i in range(data_size):
        # 28*28 pixs
        img = data[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def split_data(data_set, data_target, k):
    """
        Description:
            1) Split the data set to train and test.
            2) according to k to split the data set to train and test
        Parameters:
            data_set: the whole data
            data_target: the whole target
            k: ratio of train data set
    """
    cate_dict = {}
    train_data = []
    test_data = []
    test_target = []
    train_target = []
    # classification
    for i, item in enumerate(data_set):
        if data_target[i] not in cate_dict:
            cate_dict[data_target[i]] = []
            cate_dict[data_target[i]].append(data_set[i])
        else:
            cate_dict[data_target[i]].append(data_set[i])
    len_dict = {}
    for key in cate_dict.keys():
        len_dict[key] = int(len(cate_dict[key])*k)
    # generate the train and test data and target for each category
    for key in len_dict.keys():
        curr_train_len = len_dict[key]
        train_data += cate_dict[key][:curr_train_len]
        test_data += cate_dict[key][curr_train_len:]
        train_target += (np.ones(curr_train_len)*key).tolist()
        test_target += (np.ones(len(cate_dict[key]) - curr_train_len)*key).tolist()
    train_data = np.array(train_data)
    train_target = np.array(train_target)
    test_data = np.array(test_data)
    test_target = np.array(test_target)
    seed = 10
    # shuffle the ndarray
    rd.seed(seed)
    rd.shuffle(train_data)
    rd.seed(seed)
    rd.shuffle(train_target)
    rd.seed(seed)
    rd.shuffle(test_data)
    rd.seed(seed)
    rd.shuffle(test_target)
    print('length of the train_data %d ' % len(train_data))
    print('length of the train_target %d ' % len(train_target))
    print('length of the test_data %d' % len(test_data))
    print('length of the test_target %d' % len(test_target))
    return train_data, train_target, test_data, test_target


def define_model():
    """Define the model."""
    model = Model()
    # the input layer
    input = Input(shape=())
    return model


if __name__ == '__main__':
    # load the minist data set
    data_set_name = 'MNIST original'
    minist_dataset = load_minist(data_set_name)
    data_set = minist_dataset['data']
    data_target = minist_dataset['target']
    k = 0.8  # 80% training set and 20% test set
    train_data, train_target, test_data, test_target = split_data(data_set, data_target, k)
    # print_minit_demo(train_data[6000:6020])
    

