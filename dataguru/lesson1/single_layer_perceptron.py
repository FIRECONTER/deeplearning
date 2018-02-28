"""
The single layer perceptron learning demo.
Description:
    each step to choose the data to train the weight, if use the random select method
    what is the different?
"""


import numpy as np
import numpy.random as rd


def sgn(arg0):
    """Hardlimites signal function."""
    return 1 if arg0 >= 0 else -1


def update_weight(step, w, x_vec, y):
    """Train the weight vector."""
    w = w + step*(y-sgn(np.sum(w*x_vec)))*x_vec
    return w


def init_weight(seed, length):
    """Init the weight vector of the network 0<Wj<1."""
    rd.seed(seed)
    sigma = 0.01
    w = sigma*rd.randn(length)
    return w


def evaluate_method(train_x, train_y, w):
    """Test the evaluate value is equal to the train value."""
    for id, item in enumerate(train_x):
        res = train_y[id] - sgn(np.sum(w*item))
        if res == 0:
            continue
        else:
            return False
    return True


def train_process(train_x, train_y):
    """Train the weight."""
    length = train_x.shape[1]
    samplelen = train_x.shape[0]
    # init random seed
    seed = 20
    count = 0
    cursor = 0
    threshold = 1000
    step = 0.1
    w = init_weight(seed, length)
    print('current init weight vector')
    print(w)
    while True:
        w = update_weight(step, w, train_x[cursor], train_y[cursor])
        count = count + 1
        cursor = cursor + 1
        if count % samplelen == 0:
            cursor = 0  # reset the cursor
            res = evaluate_method(train_x, train_y, w)
            if res:
                print('training times %d ' % count)
                return w
        if count >= threshold:
            print('train too many times and excess the threshold')
            print('no linear')
            return w


def test_func(train_x, train_y, w):
    """Test the weight is right or not."""
    for id, item in enumerate(train_x):
        res = sgn(np.sum(w*item))
        y = train_y[id]
        print('real value is %d and evaluate value is %d' % (y, res))


if __name__ == '__main__':
    train_x = [[-1, 1, -2, 0], [-1, 0, 1.5, -0.5], [-1, -1, 1, 0.5]]
    train_y = [-1, -1, 1]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # begin to train the data
    w = train_process(train_x, train_y)
    test_func(train_x, train_y, w)
