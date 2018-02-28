#-*- coding:utf8 -*-
"""
Credit problem with keras bp neural network
Credit problem with SVM classification
"""


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os
import csv
import re
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
import numpy.random as rd


def loaddata(filepath):
    """Load data file."""
    res = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        src_res = list(reader)
    for item in src_res:
        tmp = item[0].split(' ')
        res.append(tmp)
    return res


def test_srcdata(src_arr):
    """Test the shape of the src array."""
    arr = np.array(src_arr)
    print('current shape is (%s, %s)' % (arr.shape[0], arr.shape[1]))


def clean_data(src_data):
    """
    Clean the src data.
    description:
        change the str to integer
        if the str is start with letter should cat the str
        and choose the last letter as a integer
    """
    reg = '^[A-Za-z]'
    res = []
    for item in src_data:
        tmp = []
        for sub_item in item:
            if re.match(reg, sub_item):
                # choose the last letter as the value AXXX
                tmp.append(int(sub_item[len(sub_item)-1:]))
            else:
                tmp.append(int(sub_item))
        res.append(tmp)
    return res


def split_samples(cleandata):
    """Split the whole src data set get the positive samples and negative samples."""
    positive_samples = []
    negative_samples = []
    for item in cleandata:
        if item[len(item)-1] == 1:
            positive_samples.append(item.tolist())
        else:
            negative_samples.append(item.tolist())
    return positive_samples, negative_samples


def generate_train_data(positive_samples, negative_samples, seed, M, K):
    """Generate training data and test data and according to the scale random split the data set."""
    positive_num = len(positive_samples)
    negative_num = len(negative_samples)
    train_positive_num = int(positive_num*K/M)
    train_negative_num = int(negative_num*K/M)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    rd.seed(seed)
    # generate the sequence of the data set
    train_pos_seq = set()
    train_neg_seq = set()
    while True:
        tmp = rd.randint(0, positive_num)
        train_pos_seq.add(tmp)
        if len(train_pos_seq) == train_positive_num:
            print(' positvie sequence generated ')
            break
    while True:
        tmp = rd.randint(0, negative_num)
        train_neg_seq.add(tmp)
        if len(train_neg_seq) == train_negative_num:
            print(' negative sequence generated ')
            break
    print(' current positive sequence in training set ')
    print(train_pos_seq)
    print(' current negative sequence in training set ')
    print(train_neg_seq)
    for id, item in enumerate(positive_samples):
        if id in train_pos_seq:
            tmp = item[:len(item)-1]
            train_x.append(tmp)
            train_y.append(item[len(item)-1]-1)
        else:
            tmp = item[:len(item)-1]
            test_x.append(tmp)
            test_y.append(item[len(item)-1]-1)
    for id, item in enumerate(negative_samples):
        if id in train_neg_seq:
            tmp = item[:len(item)-1]
            train_x.append(tmp)
            train_y.append(item[len(item)-1]-1)
        else:
            tmp = item[:len(item)-1]
            test_x.append(tmp)
            test_y.append(item[len(item)-1]-1)
    # save the training data and test data
    # random select from the positive samples
    return train_x, train_y, test_x, test_y


def normalize_data(dataset):
    """Use sklearn.preprocessing.MinMaxScaler to normalize the data of each column."""
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    res_mat = min_max_scaler.fit_transform(np.mat(dataset).T)
    return (np.mat(res_mat).T).tolist()


if __name__ == '__main__':
    src_file_path = './data/src_data.csv'
    clean_file_path = './data/clean.csv'
    if not os.path.exists(clean_file_path):
        src_data = loaddata(src_file_path)  # matrix
        cleandata = clean_data(src_data)
        np.savetxt(clean_file_path, cleandata, delimiter=',')
    cleandata = np.loadtxt(open(clean_file_path, 'r'), delimiter=',', dtype='int32')
    # change dtype
    print('current cleandata size (%s, %s)' % (cleandata.shape[0], cleandata.shape[1]))
    positive_samples, negative_samples = split_samples(cleandata)
    # 700 positive samples and 300 negative samples
    print('current positive and negative shape (%s, %s)' % (len(positive_samples), len(negative_samples)))
    # random splite the train data set and the test data set
    first_hidden_layer_units = 30
    second_hidden_layer_units = 20
    seed = 10
    M = 2
    K = 1  # 4/5 of all the set used to train the network and 1/5 of the rest set works as a test set
    train_x, train_y, test_x, test_y = generate_train_data(positive_samples, negative_samples, seed, M, K)
    print(' current data scale is ')
    print(' training set is %s and test set is %s ' % (len(train_x), len(test_x)))
    # use sklearn preprocessing to normalization the range to (0,1)
    train_x = np.array(normalize_data(train_x))
    train_y = np.array(train_y)
    test_x = np.array(normalize_data(test_x))
    test_y = np.array(train_y)
    # use keras to build the bp neural network
    model = Sequential()
    # add the first hidden layer
    model.add(Dense(units=first_hidden_layer_units, input_dim=20, activation='sigmoid'))
    # add the second hidden layer
    model.add(Dense(units=second_hidden_layer_units, activation='sigmoid'))
    # add the output layer
    model.add(Dense(units=1, activation='sigmoid'))
    #  compile the whole neural network
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    #  train the model
    model.fit(train_x, train_y, epochs=2000, batch_size=32)
    # evaluate the model
    score = model.evaluate(test_x, test_y, batch_size=32)
    print(' current score of the bp neural net work is %.1f ' % score[1])
    #  accuracy  70% why

    # use svm to deal with the classification problem
    svcobj = svm.SVC()
    svcobj.fit(train_x, train_y)
    svcres = svcobj.predict(test_x)
    score = metrics.accuracy_score(svcres, test_y)
    print(' current score of the svm model is %s ' % score)
