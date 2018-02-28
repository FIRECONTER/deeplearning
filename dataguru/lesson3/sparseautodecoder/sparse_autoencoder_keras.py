#-*- coding:utf-8 -*-
"""
Sparse autoencoder with keras
"""

from keras.models import Sequential
from keras.layers import Dense
import tools.handle_img as hd


def training_process(train_x, train_y, hidden_num, input_dim):
    """Use auto encoder to train the neural network."""
    output_num = input_dim
    model = Sequential()
    # use multi-layer to encode the data
    model.add(Dense(units=10, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(units=hidden_num, activation='sigmoid', name='compressed_layer'))
    model.add(Dense(units=10, activation='sigmoid'))
    model.add(Dense(units=output_num, activation='sigmoid', name='uncompressed_layer'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=15000, batch_size=64)
    # save the weights of the model
    # save the model to a json file
    model_json = model.to_json()
    with open('./modelstructure/auto_encoder.json', 'w') as f:
        f.write(model_json)
    model.save_weights('./weightsdata/auto_encoder.h5')
    print(' model and weights saved ')
    # save the weights to a hdf5 file
    # we can also both save the model structure and weights data to a hdf5 file
    # 小结
    # 针对于普通的自编码器 其中如果输入数据的维度相比于压缩后的维度教大 可以添加多个隐藏层
    # 作为压缩编码的层次 逐层递减最后到压缩的维度 这样比只使用单个压缩层次 设置输出神经元
    # 为压缩的维度效果更好


def training_process_sparse(train_x, train_y, hidden_num, input_dim):
    """Add decay item to make the hidden sparse."""
    output_num = input_dim
    model = Sequential()
    model.add(Dense(units=hidden_num, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(units=output_num, activation='sigmoid'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=15000, batch_size=64)
    print(' model fit finished ')
    model_json = model.to_json()
    with open('./modelstructure/sparse_auto_encoder.json', 'w') as f:
        f.write(model_json)
    model.save_weights('./weightsdata/sparse_auto_encoder.h5')


if __name__ == '__main__':
    img_path = './srcdata/lena.bmp'
    img = hd.read_image(img_path)
    K = 4
    hidden_num = 6
    devided_data = hd.devide_block(img, K)
    print(' current block dimension sample number is %s and sample dimension is %s ' % (devided_data.shape[0], devided_data.shape[1]))
    # set the input dimension
    input_dim = K*K
    train_x = hd.normalize_sample(devided_data)
    train_y = train_x
    print(' begin to train the model ')
    # training_process(train_x, train_y, hidden_num, input_dim)
    # make the hidden layer neuron number the same as the input layer and add a sparse decay to realize
    # the sparse auto encoder
    sparse_num = K*K
    training_process_sparse(train_x, train_y, sparse_num, input_dim)
