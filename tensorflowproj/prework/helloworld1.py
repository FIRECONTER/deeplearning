# -*- coding: utf-8 -*-
"""
Hello world in tf.

This is a test file.
"""

import tensorflow as tf

if __name__ == '__main__':
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)
    print('node1 and node2')
    print(node1)
    print(node2)
    print('type of node is '+str(type(node1)))  # tensor type

    # build a session and run the computational graph
    sess = tf.Session()
    tfsessionres = sess.run([node1, node2])
    print('session run result is '+str(tfsessionres))
    print('type of the session is '+str(type(tfsessionres)))
    # node3 = tf.add(node1, node2)
    # if use the + operator instead of the tf.add and it works well
    # so we can use the + operator to instead of the tf.add function
    node3 = node1 + node2
    print('add each node result '+str(sess.run(node3)))

    # the above code will build a constant result graph.

    # placeholder use like a promise to provide a value

    point_a = tf.placeholder(tf.float32)
    point_b = tf.placeholder(tf.float32)
    # add_node = point_a + point_b  # works like the tf.add(a,b)
    # we use tf.add to instead of the + opeartor and it works well
    add_node = tf.add(point_a, point_b)
    print('use placeholder result is '+str(sess.run(add_node, {point_a: [4, 5], point_b: [5, 7]})))

    add_and_triple = add_node*3
    #  3*x + 3*y
    print('the last result is '+str(sess.run(add_and_triple, {point_a: 4, point_b: 5})))


    # some simple linear operation with tf 
