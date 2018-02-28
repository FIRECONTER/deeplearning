# -*- coding:utf-8 -*-
"""
Compress the image and uncompress the image
"""
from keras.models import model_from_json
from keras.models import Model
import tools.handle_img as hd
import numpy as np


if __name__ == '__main__':
    model_filename = './modelstructure/auto_encoder.json'
    weights_filename = './weightsdata/auto_encoder.h5'
    compressed_filename = './resdata/lena_compress.zip'
    uncompressed_filename = './resdata/lena_uncompress.bmp'
    with open(model_filename, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_filename)
    # compress the image data
    img_path = './srcdata/lena.bmp'
    img = hd.read_image(img_path)
    K = 4
    devided_data = hd.devide_block(img, K)
    input_x = hd.normalize_sample(devided_data)
    compress_model = Model(inputs=model.input, outputs=model.get_layer('compressed_layer').output)
    compress_output = compress_model.predict(input_x)
    np.savetxt(compressed_filename, compress_output, delimiter=',')
    # use the model to zip the files

    # use the whole encode model to generate the picture
    res_data = model.predict(input_x)
    res_data = hd.unnormalize_sample(res_data)
    res_img = hd.undevide_block(res_data, K)
    hd.save_image(uncompressed_filename, res_img)
