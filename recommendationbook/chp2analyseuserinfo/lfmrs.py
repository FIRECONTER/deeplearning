# -*- coding:utf-8 -*-
"""
Use LFM on current data set
Author:allocator
Time:28/12/2017
"""


import pandas as pd
import os
import numpy as np
import np.random as rd

def randomSample(data, seed):
    """Random sample to generate the positive and negative samples."""
    

if __name__ == '__main__':
    datafolder = './cleandata'
    filename = 'user_movie.csv'
    columns = ['userId', 'viewlist', 'likelist']
    inputdata = pd.read_csv(os.path.join(datafolder, filename), delimiter='|')
