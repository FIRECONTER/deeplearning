# -*- coding:utf-8 -*-
"""
The third api learning.

Description:
    1) learned how to use placeholder and Variable in tensorflow
    2) used the MSE to build the loss function
    3) learned how to change the variable and rebuild the model to get the loss result
    4) learn how to use optimizers in tf
Author:alex
Time:22/11/2017
"""
import tensorflow as tf


if __name__ == '__main__':
    # generate the linear model
    w = tf.Variable([-2.0], dtype=tf.float32)
    b = tf.Variable([-3.0], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = w*x+b
    # use mse as the loss function
    loss = tf.reduce_sum(tf.square(linear_model-y))

    # use the gradientdescent optimizer as the optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)  # use the gradientdescentoptimizer set the step
    session = tf.Session()
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()  # initialize the variables
    session.run(init)  # init the computational graph
    # set the training data
    x_train = [1, 2, 3, 4]
    y_train = [2, 4, 6, 8]
    # train the model
    for i in range(1000):
        session.run(train, {x: x_train, y: y_train})
    curr_w, curr_b, curr_loss = session.run([w, b, loss], {x: x_train, y: y_train})
    print('optimized variable is w:%s b:%s loss:%s' % (curr_w, curr_b, curr_loss))
    
    # how to use tf.estimator simplifies the mechanics? loops?
