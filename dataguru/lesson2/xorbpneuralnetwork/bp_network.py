# -*- coding:utf-8 -*-
"""
Bp neural network to solve the XOR problem.
Description:
    one hidden layer BP neural network
    hidden_num means the number of neural units in hidden layer
    output_num means the number of neural units in output layer
    array compute with np.array
    matrix compute with np.mat
Author:allocator
Time:9/1/2018
"""


import numpy as np
import numpy.random as rd
import os
import csv


def init_weightmat(seed, input_num, hidden_num, output_num):
    """Init the weight matrix."""
    rd.seed(seed)
    W = rd.rand(output_num, hidden_num+1)
    V = rd.rand(hidden_num, input_num)
    return np.mat(W), np.mat(V)


def activate_func(x):
    """Activate function in BP network and x is an array."""
    return 1/(1+np.exp(-x))


def network_error_calculate(error_list):
    """
    Calculate the whole network error value.

    Description:
        when finished the whole sample data training process
        and then calculate the whole error value with RME
    """
    return np.sqrt(np.sum(error_list)/len(error_list))


def sample_error(predict, real):
    """Calculate current sample error value."""
    error_vector = predict-real
    return np.sum([np.power(item, 2) for item in error_vector])


def train_process(train_x, train_y, seed, step, hidden_num, output_num, Emin, threshold):
    """BP training process."""
    sample_num = train_x.shape[0]
    input_num = train_x.shape[1]
    W, V = init_weightmat(seed, input_num, hidden_num, output_num)
    sample_cursor = 0
    count = 0
    while True:
        print('begin to train %d ' % count)
        E_bp = 0  # set the training error
        error_list = []
        currentsample = train_x[sample_cursor]
        hidden_net = np.array((V*(np.mat(currentsample).T)).T.tolist()[0])  #alist
        hidden_out = activate_func(hidden_net)  # list
        # put -1 in the hidden_out
        hidden_out = hidden_out.tolist()
        hidden_out.insert(0, -1)
        hidden_out = np.array(hidden_out)
        output_net = np.array((W*(np.mat(hidden_out).T)).T.tolist()[0])
        output_out = activate_func(output_net)  # a list type
        # calculate current error value of current sample data
        current_value = train_y[sample_cursor]

        # add the current error data to the error list
        error_list.append(sample_error(output_out, current_value))

        # calculate the deltaW and deltaV
        #
        output_error_signal = (current_value - output_out)*(np.ones(len(output_out)) - output_out)*output_out
        hidden_error_signal = np.array(((W.T*output_error_signal).T).tolist()[0])*hidden_out*(np.ones(len(hidden_out))-hidden_out)
        # use matrix calculate the hidden_error_signal

        deltaW = step*(np.mat(output_error_signal).T)*np.mat(hidden_out)
        deltaV = step*(np.mat(hidden_error_signal[1:]).T)*np.mat(currentsample)
        # count +1

        W = W + deltaW
        V = V + deltaV
        sample_cursor = sample_cursor+1
        count = count + 1
        if sample_cursor == sample_num:
            # begin to calculate all the errors in the network
            sample_cursor = 0
            E_bp = network_error_calculate(error_list)
            print('current network error % s and emin is %s ' % (E_bp, Emin))
            if E_bp < Emin:
                print('training the net work times % s' % count)
                return W, V
        if count >= threshold:
            print('trainig to many times % s' % count)
            return None


def load_data():
    """Load data."""
    return None


def predict(sample, W, V):
    """Use the W and V to calculate the result."""
    hidden_out = activate_func(np.array((V*(np.mat(sample).T)).T.tolist()[0]))
    hidden_out = hidden_out.tolist()
    hidden_out.insert(0, -1)
    hidden_out = np.array(hidden_out)
    res = activate_func(np.array(((W*(np.mat(hidden_out).T)).T).tolist()[0]))
    print('current sample prediction is %s ' % res[0])
    return 1 if res[0] >= 0.5 else 0


def evaluate(train_X, train_y, W, V):
    """Calculate the accurary."""
    sample_num = len(train_X)
    count = 0
    for id, item in enumerate(train_X):
        res = predict(item, W, V)
        if res == train_y[id]:
            count = count + 1
    return count/sample_num


if __name__ == '__main__':
    # try to use nor on the bp neural network
    train_x = np.array([[-1, 1, 1], [-1, 1, 0], [-1, 0, 1], [-1, 0, 0]])
    train_y = np.array([0, 1, 1, 0])
    seed = 10
    step = 0.05
    hidden_num = 5
    output_num = 1
    Emin = 0.02
    threshold = 2000000
    if not (os.path.exists('./W.csv') and os.path.exists('./V.csv')):
        W, V = train_process(train_x, train_y, seed, step, hidden_num, output_num, Emin, threshold)
        print('current traing w')
        print(W)
        print('current trainint v')
        print(V)
        with open('./W.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(W.tolist())
        with open('./V.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(V.tolist())
    # train the neural network
    W = np.loadtxt(open('./W.csv', 'r'), delimiter=',')
    V = np.loadtxt(open('./V.csv', 'r'), delimiter=',')
    print('loaded W')
    print(W)
    print('loader V')
    print(V)
    W = np.mat(W)
    V = np.mat(V)
    for id, item in enumerate(train_x):
        res = predict(item, W, V)
        print('current res is %s ' % res)
    # the XOR problem can be solved by the BP neural network
    accurary = evaluate(train_x, train_y, W, V)
    print('the accurary is %.02f %%' % (accurary*100))
