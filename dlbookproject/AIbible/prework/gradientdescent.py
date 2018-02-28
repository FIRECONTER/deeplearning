# -*- coding:utf-8 -*-
"""
Gradient Descent alogrithm.

Description:simple gradient descent alogrithm
Author:alex
Time:20/11/2017
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal()
b = np.random.normal()
c = np.random.normal()


def h(x):
    """Set the hypothesis function."""
    return a*x[0]+b*x[1]+c


if __name__ == '__main__':
    # real function
    # y = 2*x1+3*x2+5
    step = 0.001  # set the step of the gradientDescent method
    x_train = np.array([[1, 2], [4, 5], [6, 8], [10, 11]])
    y_train = np.array([2*item[0]+3*item[1]+5 for item in x_train])
    test_x = np.array([[3, 5], [6, 7], [8, 9], [10, 11]])
    real_test = np.array([2*item[0]+3*item[1]+5 for item in test_x])
    for i in range(1000):
        sum_a = 0
        sum_b = 0
        sum_c = 0
        for x, y in list(zip(x_train, y_train)):
            sum_a = sum_a + step*(y-h(x))*x[0]
            sum_b = sum_b + step*(y-h(x))*x[1]
            sum_c = sum_c + step*(y-h(x))*1
        a = a + sum_a
        b = b + sum_b
        c = c + sum_c
        plt.plot([h(item) for item in test_x])
print('last a:%s b:%s c:%s ' % (a, b, c))
plt.plot(real_test)
plt.show()
