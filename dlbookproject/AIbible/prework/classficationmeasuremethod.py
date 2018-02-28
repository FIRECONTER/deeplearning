# -*- coding:utf-8 -*-
"""
Classification measure standard overview.

Description:realize all the classification measure standard.
Author:alex
Time:27/11/2017
"""

import numpy as np
import matplotlib.pyplot as plt


def errorrate(train_y, real_y):
    """Calculate the error rate of the classification result."""
    datalen = len(train_y)
    num = 0
    for t_y, r_y in list(zip(train_y, real_y)):
        if t_y != r_y:
            num = num+1
    return num/datalen


def accuraryrate(train_y, real_y):
    """Calculate the accurary rate of the classification result."""
    return 1-errorrate(train_y, real_y)


def precisionrate(train_y, real_y):
    """
    Calculate the precision rate of the classification result.

    Description:
        1) TP: sample is true and prediction is true
        2) FP: sample is false and prediction is true
        3) TN: sampe is false and prediction is false
        4) FN: sample is true and prediction is false
        calculate equation TP/(TP+FP)
    """
    TP = 0
    FP = 0
    for t_y, r_y in list(zip(train_y, real_y)):
        if t_y == 1:  # use 1 present the positive sample and 0 present the negative sample
            if r_y == 1:
                TP = TP+1
            else:
                FP = FP+1
    return TP/(TP+FP)


def recallrate(train_y, real_y):
    """
    Calculate the recall rate of the classification result.

    Description:
        calculate equation: TP/(TP+FN)
    """
    TP = 0
    ALLP = 0  # ALLP is TP+FN means all the positive sample in the training data set
    ALLP = np.sum([item for item in real_y if item == 1])
    for t_y, r_y in list(zip(train_y, real_y)):
        if t_y == 1:
            if r_y == 1:
                TP = TP+1
    return TP/ALLP


def drawPRfig():
    """
    Draw the P-R figure of the classification result.

    Description:
        1)how to use the sample data to draw the PR figure
        2)different model with different PR figure and how to choose the model
        3)if the A line contains B line that means the performance of model A is better than B
        4) use the break-even point to measure the performance of model A and model B
    Appendence:

    """
    pass


def F1measure(p, r):
    """
    Calculate F1 measure.

    Description:
        equation:1/F1 = (1/2)*(1/p+1/r)
        F1 = 2*pr/(p+r)
    """
    return 2*p*r/(p+r)


def Fbetameasure(p, r, beta):
    """
    General than F1 measure and use parameter beta to rise the weight of precision and recall.

    Description:
        1) if beta = 1 it means the F1 measure method
        2) if beta >1 it means that the recall weight is much attention than the precision weight
        3) if beta <1 it means that the precision weight is much attention than the recall weight
        equation:
        1/Fbeta = (1/(1+beta*beta)*(1/p+beta*beta/(1+r)))
        Fbeta = (1+pow(beta, 2))*p*r/(pow(beta, 2)*p+r)
    """
    return (1+pow(beta, 2))*p*r/(pow(beta, 2)*p+r)


def macroF1(plist, rlist):
    """
    Calculate the macroF1 of the model.

    Description:
        if we trained the model many times or use differernt data set to train the model
        so we get different p and r with each training process. we will get a precision rate list
        and a recall rate list so we can use these list to calculate the mean measure standard. and the
        macroF1 is one of the measure method
        equation:
            meanp = sum(plist)/N
            meanr = sum(rlist)/N
            macroF1 = 2*meanp*meanr/(meanp+meanr)
    """
    meanp = sum(plist)/len(plist)
    meanr = sum(rlist)/len(rlist)
    return 2*meanp*meanr/(meanp+meanr)


def microF1(tplist, fplist, fnlist, tnlist):
    """
    Calculate the microF1 of the model.

    Description:
    Simulate to the macroF1 measure the different is how to calculate the mean precision
    and mean recall rate. In microF1 we can calculate the mean TP FP FN TN of all the list
    and get the mean p and mean r then calculate the microF1
    """
    meantp = sum(tplist)/len(tplist)
    meanfp = sum(fplist)/len(fplist)
    meanfn = sum(fnlist)/len(fnlist)
    meantn = sum(tnlist)/len(tnlist)
    tmp_p = meantp/(meantp+meanfp)
    tmp_r = meantp/(meantp+meanfn)
    return 2*tmp_p*tmp_r/(tmp_p+tmp_r)


# realize all the standard with python libs

if __name__ == '__main__':
    train_y = [1,1,1,1,0,1,0]
    real_y = [1,0,1,1,0,0,0]
    print('the error rate is %.2f %%' % (errorrate(train_y, real_y)))
    print('the accurary rate is %.2f %%' % (accuraryrate(train_y, real_y)))
    p = precisionrate(train_y, real_y)
    r = recallrate(train_y, real_y)
    print('the precision rate is %.2f %%' % p)
    print('the recall rate is %.2f %%' % r)
    print('the fi measure is %.2f %%' % F1measure(p, r))
    # calculate the parameters with lib
