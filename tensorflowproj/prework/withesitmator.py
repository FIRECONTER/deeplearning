# -*- coding:utf-8 -*-
"""
Use estimator in tensorflow to simplifies the loop process.

Description:tf.estimator can make the ML process easy
Author:alex
Time:22/11/2017
"""
import numpy as np
import tensorflow as tf
from tensorflow import estimator

if __name__ == '__main__':
    # with no computational graph and build session
    model_fn =
    params =  # Hparams of the model
    config =
    estimator_ins = estimator.Estimator(model_fun=model_fun, params=params, config=config)
