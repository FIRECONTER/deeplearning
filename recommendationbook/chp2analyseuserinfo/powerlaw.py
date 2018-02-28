# -*- coding:utf-8 -*-
"""
Power Law distribution

Description
    y = a*x**K
"""


import numpy as np
import matplotlib.pyplot as plt


def powerlawdistribution(inputdata, alpha, k):
    return sorted([alpha*np.power(item, k) for item in inputdata], reverse=True)


if __name__ == '__main__':
    testdata = [i for i in range(20)]
    res = powerlawdistribution(testdata, 0.8, -1)
    plt.figure(figsize=(8, 8))
    plt.plot(testdata, res, 'r', label='powerlawdistribution')
    plt.xlabel('input points')
    plt.ylabel('powerlaw result')
    plt.legend(loc='upper right')
    plt.show()
