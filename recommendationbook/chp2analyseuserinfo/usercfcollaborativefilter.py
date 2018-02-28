# -*- coding:utf-8 -*-
"""
Collaborativefilter on the movielens web site.

Description:
    1) use the data set from movielens to realize the collaborative filter
Author:alex
Time:27/11/2017
"""

import os
import numpy as np
import pandas as pd
import datetime
import scipy.sparse as sp
from tools.tools import *


def splitdata(sampledata, M, K, seed):
    """
    Generate the cross validation training data and validation data.

    Description:
        get M paire data for training and validation
        cross validation
    Parameter:
        sampledata:
            structure:[userid, viewlist, likelist]
        M:
        K:
        seed:
    """
    np.random.seed(seed)
    validation_data = []
    for item in sampledata:
        if np.random.randint(0, M) == K:
            validation_data.append(item)
    return validation_data


def getrecommendation(curruser, userids, viewlist, W, K):
    """
    Core function to get the best probability likely items for the user.

    Description:
        get the recommendation item list of the user
    Parameters:

    """
    currid = userids.index(curruser)
    relateduserindex = W[currid].toarray()[0].tolist()[:K]
    # use the set to contain all the prossible items
    # generate the item_user table
    itemuniontable = {}
    for i in relateduserindex:
        curr_items = viewlist[i]
        for item in curr_items:
            if item not in itemuniontable:
                itemuniontable[item] = []
            itemuniontable.append(userids[i])
    # final rank list
    ranklist = []
    for key in itemuniontable.keys():
        curritemname = key
        curritemrank = 0
        for i in itemuniontable[key]:
            curritemrank = curritemrank+W[currid, i]
        ranklist.append([curritemname, curritemrank])
    return ranklist


def recallrate(test_set, userids, viewlist, W, K):
    """
    Calculate the recall rate of the module in collaborative filter.

    Descriotion:
        sum(R(u)&T(U))/sum(T(u))
    """
    hitnum = 0
    allnum = 0
    for userinf in test_set:
        real_like = userinf[2]  # real like list
        curruser = userinf[0]
        rank = getrecommendation(curruser, userids, viewlist, W, K)
        # rank is a sorted list contains the item and the probability
        for item, pui in rank:
            if item in real_like:
                hitnum = hitnum+1
        # add the real like items
        allnum = allnum+len(real_like)
    return hitnum/allnum


def precisionrate(test_set, userids, viewlist, W, K):
    """
    Calculate the precision rate of the module in collaborative filter.

    Description:
        sum(R(u)&T(u))/sum(R(u))
        general precision defination
    """
    hitnum = 0
    allnum = 0
    for userinf in test_set:
        curruser = userinf[0]
        real_like = userinf[2]
        rank = getrecommendation(curruser, userids, viewlist, W, K)
        allnum = allnum+len(rank)
        for item, pui in rank:
            if item in real_like:
                hitnum = hitnum+1
    return hitnum/allnum


def coveragerate(test_set, userids, viewlist, W, K):
    """
    Galculate the coverage rate.

    Description:
        sum(all(recommendationtype))/allitems
    """
    recomitem_set = set()
    allitem_set = set()
    for userinf in test_set:
        curruser = userinf[0]
        for item in userinf[1]:
            allitem_set.add(item)
        rank = getrecommendation(curruser, userids, viewlist, W, K)
        for item, pui in rank:
            recomitem_set.add(item)
    return len(recomitem_set)/len(allitem_set)


def popularity(test_set, userids, viewlist, W, K):
    """
    Calculate the popularity.

    Description:
        sum(log(popularity))/N focus on the recommendation items
    """
    for userinfo in test_set:
        # calculate the popularity of each item in train_set
        item_popularity = dict()
        curruser = userinfo[0]
        for item in userinfo[1]:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] = item_popularity[item]+1
        n = 0  # calculate all the recommendation items
        po_sum = 0
        rank = getrecommendation(curruser, userids, viewlist, W, K)
        for item, pui in rank:
            po_sum = po_sum+np.log(1+item_popularity[item])
            # calculate each recommendation item's popularity
            n = n+1
        return po_sum/n


def usersimilarity(userid, viewlist):
    """
    Calculate the similarity matrix of the user set.

    Description:
        each value in the matrix means the similarity between the two persons
        there are to many methods to calculate the similarity between each user
        we can choose the jaccard equation and the cosin similarity equation:
            1)jaccard equation:
                Wuv = |N(u)&N(v)|/|N(u)+N(v)|
            2)consin equation:
                Wuv = |N(u)&N(v)|/sqrt(|N(u)||N(v)|)
    appendence:
        the matrix is to huge we should use sparse matrix to save the matrix
    """
    # generate the item user table
    user_len = len(userid)
    item_user = {}

    start_time = datetime.datetime.now()
    total_start_time = start_time
    print('start to generate the item_user table %s ' % start_time)
    for i in range(user_len):
        curritemlist = viewlist[i]
        for item in curritemlist:
            if item not in item_user:
                item_user[item] = []
            item_user[item].append(userid[i])
    end_time = datetime.datetime.now()
    print('finish to generate the item_user table and cost time is %s' % (end_time-start_time))
    # generate the sparse matrix
    sparse_mat = sp.csr_matrix((user_len, user_len), dtype=np.uint8)
    start_time = datetime.datetime.now()
    print('begin to generate the sparse matrix %s ' % start_time)
    for key in item_user.keys():
        # userid list of current item
        idlist = [userid.index(item) for item in item_user[key]]
        for i in idlist:
            for j in idlist:
                if i == j:
                    continue
                if sparse_mat[i, j] == 0:
                    sparse_mat[i, j] == 1
                else:
                    sparse_mat[i, j] = sparse_mat[i, j]+1
    end_time = datetime.datetime.now()
    print('finish generate the sparse matrix and cost time is %s ' % (end_time-start_time))
    # finally calculate the similarity between each user
    # get all the no-zero elements in the sparse matrix
    # should change to the coo_matrix to get row and col
    rowlist = sparse_mat.tocoo().row
    collist = sparse_mat.tocoo().col
    start_time = datetime.datetime.now()
    print('final calculate the similarity sparse matrix %s ' % start_time)
    for i in rowlist:
        for j in collist:
            len_u = len(viewlist[i])
            len_v = len(viewlist[j])
            sparse_mat[i, j] = sparse_mat[i, j]/(np.sqrt(len_u*len_v))
    end_time = datetime.datetime.now()
    total_end_time = end_time
    print('finally finish the similarity matrix %s ' % (end_time-start_time))
    print('the total cost time is %s ' % (total_end_time-total_start_time))
    return sparse_mat


if __name__ == '__main__':
    # get the data file path
    SEED = 10
    M = 8
    datafolder = './cleandata'
    filename = 'user_movie.csv'
    similarity_matrix_filename = 'user_similarity.npy'
    columns = ['userId', 'viewlist', 'likelist']
    inputdata = pd.read_csv(os.path.join(datafolder, filename), delimiter='|')
    userId = list(inputdata['userId'].values)
    viewlist = [strlisttolist(item) for item in list(inputdata['viewlist'].values)]
    likelist = [strlisttolist(item) for item in list(inputdata['likelist'].values)]
    srcdata = list(zip(userId, viewlist, likelist))
    print('current data information')
    print(len(srcdata))
    # begin to train the similarity matrix
    if not os.path.exists(os.path.join(datafolder, similarity_matrix_filename)):
        W = usersimilarity(userId, viewlist)
        print('begin to save the user similarity matrix')
        np.save(os.path.join(datafolder, similarity_matrix_filename), W)
        print('save the user similarity matrix ok ')
    else:
        W = np.load(os.path.join(datafolder, similarity_matrix_filename))
    # begin to train the model
    for i in range(M):
        currval_data = splitdata(srcdata, M, i, SEED)
        break
    # fist split the data
