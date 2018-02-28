# -*- coding:utf-8 -*-
"""
This is the helloworld2 file.

Description:still test on the tf framework
Author:alex
Time:21/11/2017
"""


import tensorflow as tf

if __name__ == '__main__':
    # Variable how to use
    # when define the Variable they are not initialized
    # What is the different between variable and placeholder
    # Variable is the parameter of the model and placeholder is the input data
    w = tf.Variable([-0.3], dtype=tf.float32)
    b = tf.Variable([-0.4], dtype=tf.float32)
    x = tf.placeholder(tf.float32)  # given a x as an input tensor
    linear_model = w*x+b  # set the lienar model -0.3x-0.4
    sess = tf.Session()
    init = tf.global_variables_initializer()  # init the global variables
    sess.run(init)
    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))  # simple linear_model and the result

    # write a cost function to evaluate the linear model use mse as the lost function

    # create another linear model 2*x + 3
    w = tf.Variable([2.0], dtype=tf.float32)
    b = tf.Variable([3.0], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = w*x+b
    init = tf.global_variables_initializer()
    sess.run(init)  # init the computational graph
    y = tf.placeholder(tf.float32)  # given the real data
    square_res = tf.square(linear_model-y)  # get the square data of each predict data and real data
    loss = tf.reduce_sum(square_res)
    x_train = [1, 2, 3, 4]
    y_train = [3*item+4 for item in x_train]
    print('loss is '+str(sess.run(loss, {x: x_train, y: y_train})))  # loss is zero

    # Variable can be changed by tf.assign and still should run sess.run([var2, var2]) to change the value

    print('change the variable')
    fixW = tf.assign(w, [3.0])
    fixB = tf.assign(b, [4.0])
    sess.run([fixW, fixB])
    print('optimized variables loss result is '+str(sess.run(loss, {x: x_train, y: y_train})))
