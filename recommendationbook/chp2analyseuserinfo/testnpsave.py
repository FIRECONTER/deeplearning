# -*- coding:utf-8 -*-
"""
Test the np.save and np.load function dealing with the sparse matrix.

Description:
    np.save(filename, matrix) np.load(filename)
    the filename should be XXX.npy
Author:alex
Time:05/12/2017
"""


import numpy as np
import scipy.sparse as sp
import os


if __name__ == '__main__':
    dirfolder = os.path.join('./', 'testsave')
    if not os.path.exists(dirfolder):
        os.makedirs(dirfolder)
    t_sp1 = sp.csr_matrix((2, 2), dtype=np.uint8)
    t_sp1[1, 1] = 10
    # save the matrix
    print('current matrix')
    print(t_sp1)
    np.save(os.path.join(dirfolder, 'csr_matrix.npy'), t_sp1)
    t_sp2 = np.load(os.path.join(dirfolder, 'csr_matrix.npy'))
    print(' the load matrix ')
    print(t_sp2)
