# -*- coding:utf-8 -*-
"""
Description:
    1) 手写一个简单的卷积神经网络 并且使用minist 数据集合进行训练完成图像分类计算
"""
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt


def conv1D(input, kernel):
    """
        Description:
            1) Simple conv1D layer.
    """
    pass


def pooling(input, step):
    """Simple pooling layer."""
    pass


def relu(input):
    """Define relu function."""
    return np.array([max(0, item) for item in input])


if __name__ == '__main__':
    pass