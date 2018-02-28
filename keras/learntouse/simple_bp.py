# -*- coding:utf-8 -*-
"""
Bp neural network with keras
do the XOR simple classification problem
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


if __name__ == '__main__':
    #  training data set
    train_x = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    train_y = np.array([0, 1, 1, 0])
    # model weight matrix save file path
    #  build the whole layer model
    model = Sequential()
    #  add the hidden layer
    model.add(Dense(units=4, input_dim=2, activation='tanh'))
    #  add the output layer
    model.add(Dense(units=1, activation='tanh'))
    #  compile the whole neural network loss function and optimizer method
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    #  mini-batch gradient descent
    model.fit(train_x, train_y, epochs=10000, batch_size=4)
    score = model.evaluate(train_x, train_y, batch_size=4)
    print('current score is %s ' % score)
    print(type(score))
    # predict
    # evaluate
