# -*- coding:utf-8 -*-
"""
Description:
    1) 尝试使用keras 中的conv相关层的API做实验
    2) 更深度的熟悉keras API 从各个角度去分析模型
"""
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

def simple_mlp():
    """Simple mlp."""
    input = Input(shape=(10,))
    hidden_1 = Dense(units=5, activation='relu')(input)
    hidden_2 = Dense(units=3, activation='relu')(hidden_1)
    output = Dense(units=1, activation='relu')(hidden_2)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    return model


def conv1d_model():
    """Simple conv1D layer"""
    # dimention 10 vector
    input = Input(shape=(10,4))
    # filters 决定了卷积核的数目 kernel_size 卷积核大小
    # strides 每一个卷积核的步长值 注意兼容性
    output = Conv1D(filters=1, kernel_size=2)(input)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # curr_simple_mlp = simple_mlp()
    # print(curr_simple_mlp.summary())
    # the conv1d layer info
    curr_model = conv1d_model()
    print(curr_model.summary())
