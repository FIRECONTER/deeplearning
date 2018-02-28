# -*- coding:utf-8 -*-
"""
深度前馈神经网络解决XOR问题.

@description:
@author:alex
@time:17/11/2017
"""
import math
import numpy as np


def costfunction(tar, pre):
    """Use the mse as the cost function."""
    arrlen = len(tar)
    errorlist = [
        math.pow((item[0]-item[1]), 2) for item in list(zip(tar, pre))
    ]
    return sum(errorlist)/arrlen


def normalequation(x, y):
    """Calculate the normal equation and return the theta."""
    X = np.mat(x)
    Y = (np.mat(y)).T
    return (X.T*X).I*X.T*Y


def gradientdescent(train_x, train_y, step, lopnum):
    """Calculate the theta with gradientdescent."""
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    lentheta = train_x.shape[1]
    theta = [np.random.normal() for i in range(lentheta)]  # ndarray type

    def h(x):
        """Set the hypothesis function."""
        return np.sum(theta*x)
    for i in range(lopnum):
        step_theta = np.zeros(lentheta)  # set the shape of the step_theta
        for x, y in list(zip(train_x, train_y)):
            for i in range(lentheta):
                step_theta[i] = step_theta[i]+step*(y-h(x))*x[i]
        theta = theta + step_theta
    return list(theta)


if __name__ == '__main__':
    tar = [[1, 1], [1, 0], [0, 0], [0, 1]]
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]]
    Y = [0, 1, 0, 1]
    theta = normalequation(X, Y)
    print('use normal equation')
    print('current theta is '+str(theta))  # 0.5 0 0
    print('use gradientdescent ')
    theta = gradientdescent(X, Y, 0.01, 10000)
    print('gradientdescent to get theta '+str(theta))

    # use deepfeedforward network
    # define the input function  h(X) = W*x+C
    # define the activate function ReLU g(h) = max{0,h(X)}
    # define the output function f2(g) = w*g+b
    src_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
    src_out = [0, 1, 1, 0]
    W = [[1, 1], [1, 1]]
    w = [1, -2]
    c = [0, -1]

    # change the array to matrix
    X = np.mat(src_input)
    W = np.mat(W)
    w = np.mat(w).T
    C = np.mat(c)

    def f1(X, W, C):
        """Generate the input layer."""
        return X*W+C

    f1_out = f1(X, W, C)
    print('before the activate function result is '+str(f1_out))

    def h(X):
        """Genrate the activate layer."""
        for i, item in enumerate(X):
            res_item = [np.max([0, j]) for j in item[0].tolist()[0]]
            X[i] = res_item
        return X
    h_out = h(f1_out)
    print('after activate function result is '+str(h_out))

    def f2(X, w, b):
        """Set the output layer."""
        return ((X*w+b).T).tolist()[0]

    final_out = f2(h_out, w, 0)
    print('the final out is '+str(final_out))

    def getaccurary(train_y, real_y):
        """Generate the accuary of the deepfeedforward network."""
        num = 0
        for i in range(len(train_y)):
            if train_y[i] != real_y[i]:
                num = num+1
        return (1-(num/len(train_y)))*100

    print("The accurary is %.2f%%" % getaccurary(final_out, src_out))


    # use gradientdescent method
