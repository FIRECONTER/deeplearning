# -*- coding:utf-8 -*-
"""
Draw ROC and PR figure with limited points.

Description:
    1) Use the limited points to simulate the ROC and PR figure.
    2) The simulate methods are the same the difference is that
        PR calculate the P R and ROC calculate the TPR FPR
Author:alex
Time:28/11/2017
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def getTPRFPR(id, psize, nsize):
    """Generate the (FPR, TPR) point of each sample."""
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    if id < psize:
        TP = id+1
        FN = psize-TP
        FP = 0
        TN = nsize-FP
        return [FP/(FP+TN), TP/(TP+FN)]
    else:
        TP = psize
        FN = 0
        FP = id-psize+1
        TN = nsize-FP
        return [FP/(FP+TN), TP/(TP+FN)]


def generatePR(id, psize, nsize):
    """
    Generate the PR figure.

    The experiment method is the same with the ROC figure
    The difference is the calculate standard
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    if id < psize:
        TP = id+1
        FN = psize-TP
        FP = 0
        TN = nsize-FP
        return [TP/(TP+FN), FP/(FP+TN)]
    else:
        TP = psize
        FN = psize-TP
        FP = id-psize+1
        TN = nsize-FP
        return [TP/(TP+FN), FP/(TP+TN)]


if __name__ == '__main__':
    np.random.seed(10)  # set the random set
    # generate 100 uniform distribution data as the prediction result
    # each value in the list means the probability to be the positive samples
    pnum = 50  # positive sample number the first 50 samples
    nnum = 50  # negative sample number the last 50 samples
    samples = sorted(list(np.random.rand(100)), reverse=True)
    # print(samples)
    rocpoints = []
    prpoints = []
    for id, item in enumerate(samples):
        rocpoints.append(getTPRFPR(id, 30, 70))
        prpoints.append(generatePR(id, 30, 70))
    print(' the roc figure points ')
    print(rocpoints)
    print(' the pr figure points ')
    print(prpoints)
    # draw the ROC figure
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    x_axis, y_axis = list(zip(*rocpoints))
    plt.plot(x_axis, y_axis, 'r', label='roc curve')
    plt.title(' the roc figure ')
    plt.xlabel(' FPR ')
    plt.ylabel(' TPR ')
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.xticks(x_axis)
    plt.legend(loc='best')
    plt.subplot(122)
    x_axis, y_axis = list(zip(*prpoints))
    plt.plot(x_axis, y_axis, 'r', label='pr curve')
    plt.title(' the pr figure ')
    plt.xlabel(' R(recall) ')
    plt.ylabel(' P(precision) ')
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.xticks(x_axis)
    plt.legend(loc='best')
    plt.show()

    # this is a 100% right classifier
