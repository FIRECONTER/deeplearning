# -*- coding:utf-8 -*-
"""
Test the double random.

Description:
    random array with random length
    wrap function set the seed
    rd.seed(10)
Author:alex
Time:30/11/2017
"""

import numpy.random as rd


if __name__ == '__main__':
    rd.seed(10)
    lenarr = [20, 30, 40, 50, 60]
    for item in lenarr:
        tmp = rd.randint(1, item+1)
        size = int(tmp/3) if tmp > 2 else tmp
        currlist = rd.randint(0, item, size=size)
        print('current list')
        print(currlist)
