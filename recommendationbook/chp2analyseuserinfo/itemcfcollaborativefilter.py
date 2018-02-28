# -*- coding:utf-8 -*-
"""
ItemCF collaborative filter algorithm.

Description:
    Focus on the user informaton to realize the item cf algorithm about
    recommendation system
Author:alex
Time:05/12/2017
"""


import numpy as np
import os
import scipy.sparse as sp
import pandas as pd
import datetime


def strlisttolist(item):
    """Change the strlist to list."""
    tmp_list = item[1:len(item)-1].split(',')
    return [int(item) for item in tmp_list]


def randomsample(sampledata, M, K, seed):
    """
    Random sampling the data set.

    Description:
        current data set is huge
    """
    np.random.seed(seed)
    sampleset = []
    for item in sampledata:
        if np.random.randint(0, M) <= K:
            sampleset.append(item)
    return sampleset


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


def getrecommendation(curruser, userids, viewlist, itemlist, W, K):
    """
    Core function to get the best probability likely items for the user.

    Description:
        according to the user's history information and choose one item which k similar
        item contains the history items and calculate the similarity
    Parameters:
    """
    currindex = userids.index(curruser)
    # calculate the recommendation items
    curritemlist = viewlist[currindex]
    recomlist = []
    for id in range(len(itemlist)):
        # K similar items id
        recom_ids = sorted(W[id].toarray()[0], reverse=True)[:K]
        recom_list = [itemlist(curid) for curid in recom_ids]
        # current view list
        crossset = set(viewlist[currindex]) & set(recom_list)
        if len(crossset) == 0:
            continue
        else:
            pui = 0
            for item in list(crossset):
                pui = pui+W[id, itemlist.index(item)]
            recomlist.append([itemlist[id], pui])
    return recomlist


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


def itemsimilarity(userid, viewlist):
    """
    Calculate the item similarity matrix of the user set.

    Description:
        each value in the matrix means the similarity between the two persons
        there are to many methods to calculate the similarity between each user
        we can choose the jaccard equation and the cosin similarity equation:
            1)jaccard equation:
                Wuv = |N(i)&N(j)|/|N(i)+N(j)|
            2)consin equation:
                Wuv = |N(i)&N(j)|/sqrt(|N(i)||N(j)|)
    appendence:
        the matrix is to huge we should use sparse matrix to save the matrix
    """
    # generate the item user table
    user_len = len(userid)
    item_user = {}
    start_time = datetime.datetime.now()
    total_start_time = start_time
    print('start to generate the user_item table %s ' % start_time)
    for i in range(user_len):
        curritemlist = viewlist[i]
        for item in curritemlist:
            if item not in item_user:
                item_user[item] = []
            item_user[item].append(userid[i])
    end_time = datetime.datetime.now()
    # get the length of the item
    item_len = len(item_user.keys())
    # item list
    item_list = list(item_user.keys())
    print('finish to generate the item_user table and cost time is %s' % (end_time-start_time))
    # generate the sparse matrix
    sparse_mat = sp.lil_matrix((item_len, item_len), dtype=np.uint16)
    start_time = datetime.datetime.now()
    print('begin to generate the sparse matrix %s ' % start_time)
    for item in viewlist:
        for i in item:
            for j in item:
                if i == j:
                    continue
                else:
                    i_id = item_list.index(i)
                    j_id = item_list.index(j)
                    sparse_mat[i_id, j_id] = sparse_mat[i_id, j_id]+1
    # get the no-zero item
    rowlist = sparse_mat.tocoo().row
    collist = sparse_mat.tocoo().col
    for i in rowlist:
        for j in collist:
            i_len = len(item_user[item_list[i]])
            j_len = len(item_user[item_list[j]])
            sparse_mat[i, j] = sparse_mat[i, j]/np.sqrt(i_len*j_len)
    total_end_time = datetime.datetime.now()
    print(' the cost to generate item similarity matrix is %s ' % (total_end_time-total_start_time))
    return sparse_mat


if __name__ == '__main__':
    # get the data file path
    SEED = 10
    M = 8
    datafolder = './cleandata'
    filename = 'user_movie.csv'
    similarity_matrix_filename = 'item_similarity.npy'
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
        W = itemsimilarity(userId, viewlist)
        print('begin to save the item similarity matrix')
        np.save(os.path.join(datafolder, similarity_matrix_filename), W)
        print('save the item similarity matrix ok ')
    else:
        W = np.load(os.path.join(datafolder, similarity_matrix_filename))
    # begin to train the model
    for i in range(M):
        currval_data = splitdata(srcdata, M, i, SEED)
        break
    # fist split the data
