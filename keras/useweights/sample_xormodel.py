#-*- coding:utf-8 -*-
"""
Generate the model of xor with mlp
"""
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
import numpy as np


if __name__ == '__main__':
    pass
    # 如果需要以中间的隐层的输出作为输入以模型最终层的输出作为输出 此时
    # 使用函数式构建新的模型是无法实现的 会出现链接错误 此种情况下可能只能够获取
    # 已经fit好的模型中的权重计算最终的输出
    # 以计算异或为例子 查看每一层get_weight 能够获取到什么
    train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_y = np.array([0, 1, 1, 0])
    model = Sequential()
    model.add(Dense(units=4, input_dim=2, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=4000)
    score = model.evaluate(train_x, train_y)
    print(' current score loss %s and acc %s ' % (score[0], score[1]))
    json_str = model.to_json()
    with open('./model_structure.json', 'w') as f:
        f.write(json_str)
    model.save_weights('./model_weights.h5')
    print(' model structure and weights saved ')
