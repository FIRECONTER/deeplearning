# -*- coding:utf-8 -*-
"""
Description: try to use h5py to deal with the hdf5 file
Author: allocator
"""
import numpy as np
import h5py
import os


if __name__ == '__main__':
    data_file = '../data/img_feature/image_features.h5'
    f = h5py.File(data_file, 'r')
    current_feature = f['img_1']
    print(current_feature)
    print(current_feature.dtype)
    print(current_feature.shape)
    # convert the h5 data set to numpy np.array(dataset)
    np_feature = np.array(current_feature)
    print(len(np_feature[0]))