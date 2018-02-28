#-*- coding:utf-8 -*-
"""
Sample hopfield neural network
"""
import numpy as np
import numpy.random as rd
import copy


def generate_weights(samples):
    """Generate the weights for the hopfield neural network and bias is zero."""
    dim = samples.shape[1]
    I_mat = np.mat(np.eye(dim, dtype=int))
    weights = np.mat(np.zeros((dim, dim), dtype=int))
    for item in samples:
        tmp_mat = np.mat(item)
        tmp_mat = tmp_mat.T*tmp_mat
        tmp_mat = tmp_mat-tmp_mat[0, 0]*I_mat
        weights = weights + tmp_mat
    return weights


def activation_func(x):
    """Use the sgn as activation function."""
    return 1 if x >= 0 else -1


def calculate_process(data, weights, seed):
    """Use async mod to update the status of the net work."""
    count = 0
    res = copy.deepcopy(data)
    threshold = 20000
    neurons = weights.shape[0]
    rd.seed(seed)
    while True:
        current_node = rd.randint(neurons)
        # calculate all the updated data if X(t+1)=f(WX(t)) the state
        tmp_list = (weights*(np.mat(res).T)).T.tolist()[0]
        tmp_res = [activation_func(item) for item in tmp_list]
        if tmp_res == res:
            print(' calculate times is %s ' % count)
            return res
        else:
            res[current_node] = tmp_res[current_node]
        count = count + 1
        if count >= threshold:
            print(' calculate to many times ')
            return res


if __name__ == '__main__':
    attractors = np.array([[-1, 1, -1], [1, -1, 1]], dtype=int)
    test_data = [-1, -1, -1]
    W = generate_weights(attractors)
    res = calculate_process(test_data, W, 10)
    print(' input data is ')
    print(test_data)
    print(' finally the state status is ')
    print(res)
