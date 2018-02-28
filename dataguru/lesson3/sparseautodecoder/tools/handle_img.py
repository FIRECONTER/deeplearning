# -*- coding:utf-8 -*-
"""
Handle the image
"""
from scipy import misc
import numpy as np


def zip_block(block_img):
    """Zip the image block to a vector."""
    rows = block_img.shape[0]
    cols = block_img.shape[1]
    zip_vector = []
    for i in range(cols):
        zip_vector = zip_vector + block_img[:, i].tolist()
    return zip_vector


def unzip_block(vector, K):
    """Unzip the vector to the image block."""
    block_dim = int(len(vector)/K)
    block_img = np.zeros((block_dim, block_dim), dtype='uint8')
    for i in range(block_dim):
        block_img[: i] = vector[i*K: (i+1)*K]
    return block_img


def devide_block(img, K):
    """Devide the image to a matrix and each column contains the sub pic."""
    src_row = img.shape[0]
    src_col = img.shape[1]
    devided_row = int(src_row*src_col/(K*K))
    devided_col = K*K
    resize_row = int(src_row/K)
    resize_col = int(src_col/K)
    # each sub block of the picture and zip the sub block to a vector axis=0(row)
    devided_img = np.zeros((devided_row, devided_col), dtype='uint8')
    for i in range(resize_row):
        for j in range(resize_col):
            # the sub block img
            block_img = img[i*K:(i+1)*K, j*K:(j+1)*K]
            # zip the block
            zip_vector = zip_block(block_img)
            devided_img[i*resize_col+j, :] = zip_vector
    return devided_img


def undevide_block(res_data, K):
    """Undevide the block of the image."""
    block_num = res_data.shape[0]
    row = int(np.sqrt(block_num*K*K))
    col = row
    block_row = int(row/K)
    block_col = int(col/K)
    res_img = np.zeros((row, col), dtype='uint8')
    for i in range(block_row):
        for j in range(block_col):
            block_img = unzip_block(res_data[i*block_row+j], K)
            res_img[i*K: (i+1)*K, j*K: (j+1)*K] = block_img
    return res_img


def normalize_sample(sample_data):
    """Normalize the sample data."""
    BASE = 255
    sample_data = np.array(sample_data, dtype='float32')
    return sample_data/BASE


def unnormalize_sample(sample_data):
    """Unnormalize the sample data."""
    BASE = 255
    return np.array(np.around(sample_data*BASE), dtype='uint8')


def read_image(path):
    """Use misc to read the image."""
    img = misc.imread(path)
    return img


def save_image(path, data):
    """Save the image data."""
    misc.imsave(path, data)
