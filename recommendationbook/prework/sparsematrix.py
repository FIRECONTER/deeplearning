# -*- coding:utf-8 -*-
"""
Use the sparse matrix in scipy

Description:
    found that use the ordinary matrix to save the similarity matrix
    will cause the memorry error
    csr_matrix((data,(row_id, col_id)), shape=(m,n))
    if the array is too huge so can not use the normal
    init method to generate the sparse matrix
    so use the lil_matrix to finish this task
Author:alex
Time:04/12/2017
"""


import scipy.sparse as sp
import numpy as np


if __name__ == '__main__':
    row_id = [0,1,1,2]
    col_id = [0,1,2,1]
    arr = [2,3,4,5]
    normal_mat = sp.csr_matrix((3, 3), dtype=np.uint8)
    normal_mat[1, 2] = 20 # change the value in the sparse matrix
    print(normal_mat)
    carr = normal_mat.toarray() # change to np.ndarray
    cmat = normal_mat.todense() # change to matrix
    print('mat is')
    print(cmat)
    print('arr is ')
    print(carr)
    print('no zero list row')
    print(normal_mat.tocoo().row)
    print('no zero list col')
    print(normal_mat.tocoo().col)
    print('current sparse matrix')
    print(type(normal_mat))

    # initial a sparse matrix with no information
