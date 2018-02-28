#-*- coding:utf-8 -*-
"""
Random neural network simple boltzman
自联想型boltzmann Machine
"""
import numpy as np
import numpy.random as rd


def init_weights(seed, M, N):
    """Initial the weights of the net work."""
    rd.seed(seed)
    W = rd.randn()
    return W


def generate_distribution(train_x, train_y):
    """Generate the distribution about the pattern in the sample_data."""
    categories_list = list(set(train_y.tolist()))
    sample_len = len(train_x)
    sample_dict = dict()
    for item in categories_list:
        if item not in sample_dict:
            sample_dict[item] = 0
    for id, item in enumerate(train_x):
        current_category = train_y[id]
        sample_dict[current_category] = sample_dict[current_category] + 1
    pattern_list = [float(item[1])/sample_len for item in sorted(sample_dict.items(), key=lambda x: x[0], reverse=False)]
    return pattern_list


if __name__ == '__main__':
    M = 20  # hidden layer neuron number
    N = 10  # visible layer neuron number
