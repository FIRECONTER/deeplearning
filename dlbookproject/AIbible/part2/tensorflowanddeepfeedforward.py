# -*- coding:utf-8 -*-
"""
使用tensorflow 解决深度前馈神经网络中的XOR问题.

Description:Use the tensorflow framework to deal with the deep
    feedforward XOR problem
Author:alex
Time:23/11/2017
"""

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    src_input = []
    src_output = []
    W = []
    f1_w = tf.Variable(W)
    f1_c = tf.Variable(C)
    X = tf.placeholder(tf.int)
    f1_res = X*f1_w+f1_c

    
