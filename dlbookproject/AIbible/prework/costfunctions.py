# -*- coding:utf-8 -*-
"""
Common Cost functions.

@description:some common cost functions
@author:alex
"""
import math


def mse(tar, pre):
    """MSE mean square error."""
    arrlength = len(tar)
    tmp = list(zip(tar, pre))
    squireerrorlist = [pow((item[0]-item[1]), 2) for item in tmp]
    return sum(squireerrorlist)/arrlength


def rmse(tar, pre):
    """RMSE root mean squre error."""
    arrlength = len(tar)
    tmp = list(zip(tar, pre))
    squireerrorlist = [pow((item[0]-item[1]), 2) for item in tmp]
    return math.sqrt(sum(squireerrorlist)/arrlength)


def mae(tar, pre):
    """Mean absolute error."""
    arrlength = len(tar)
    tmp = list(zip(tar, pre))
    abserrorlist = [abs(item[0]-item[1]) for item in tmp]
    return sum(abserrorlist)/arrlength


def sdfunc(tar):
    """As the standard deviation."""
    meandata = sum(tar)/len(tar)
    errorlist = [pow((item-meandata), 2) for item in tar]
    return math.sqrt(sum(errorlist)/len(tar))


if __name__ == '__main__':
    tar = [1, 2, 3]
    pre = [0, 0, 0]
    print('MSE= '+str(mse(tar, pre)))
    print('rmse= '+str(rmse(tar, pre)))
    print('mae= '+str(mae(tar, pre)))
    print('standard deviation= '+str(sdfunc(tar)))
