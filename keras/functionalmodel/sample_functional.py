# -*- coding:utf-8 -*-
"""
Sample demo about the functional api about Keras
softmax multi-classification withe function model api and sequential model api
"""

import keras
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
import numpy.random as rd
import numpy as np


def genrate_sampledata(categories, train_num, test_num, input_dim, seed):
    """Generate the classification data."""
    rd.seed(seed)  # set the seed of the random process
    train_x = rd.random((train_num, input_dim))
    test_x = rd.random((test_num, input_dim))
    train_y = keras.utils.to_categorical(rd.randint(categories, size=(train_num, 1)), num_classes=categories)
    test_y = keras.utils.to_categorical(rd.randint(categories, size=(test_num, 1)), num_classes=categories)
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # 生成测试数据
    input_dim = 20
    train_num = 600
    test_num = 300
    categories = 10
    seed = 20
    train_x, train_y, test_x, test_y = genrate_sampledata(categories, train_num, test_num, input_dim, seed)
    inputs = Input(shape=(input_dim,))
    print(' current type of inputs is %s ' % type(inputs))
    # 当前类是tensorflow 中的Tensor 类型(张量) backend 使用tensorflow 所以使用其他的框架应该是其他框架
    # 相关的张量类型
    hidden_output = Dense(units=64, activation='relu')(inputs)
    print(' current type of hidden output %s ' % type(hidden_output))
    # Dense 本身构建的是一个keras中的Dense类型 可以看成函数 但当给输入为tensor时此时的hidden_out 输出的类型也是tensor
    output_out = Dense(units=categories, activation='softmax')(hidden_output)
    print(' current type of output content %s ' % type(output_out))
    # 同样当把keras 中的layer实例当做函数使用 输出的都是tf 中的tensor 类型
    # 一个Model 接受输入类型为tensor 输出类型为 tensor
    model_func = Model(input=inputs, output=output_out)
    print(' current type of functional model %s ' % type(model_func))
    # 使用Model 构建的model 类型
    model_func.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(' current type of functonal model after compile %s ' % type(model_func))
    model_func.fit(train_x, train_y, epochs=1, batch_size=64)
    print(' current type of functonal model after fit %s ' % type(model_func))
    # 使用模型时可以采用model.predict方式获取结果 也可以将整个模型看做一个函数model(X)
    # 接受一个输入的tensor 类型 然后输出一个tensor 类型
    # 函数式用法更加灵活与底层 而贯序式用法 会做更好的封装 输出类型更加顶层
    print(' use the functional model ')
    # model_func(train_x)
    print(test_x[0])
    #model_func_res = model_func.predict(test_x[0])
    #print(' current functional model predict res type %s and number %s ' % (type(model_func_res), model_func_res))


    # 使用贯序模型API构建的model
    print('-'*30+' compare two models '+'-'*30)
    model_se = Sequential()
    model_se.add(Dense(units=64, activation='relu', input_dim=input_dim))
    model_se.add(Dense(units=categories, activation='softmax'))
    print(' current type of sequential model %s ' % type(model_se))
    # 同样的网络结构 但是采用贯序模型构建的model 和使用函数式API构建的model类型不同
    model_se.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(' current type of sequential model after compile %s ' % type(model_se))
    # 编译不会改变模型的类型 所以两种构建方式编译得到的模型类型还是不同的
    model_se.fit(train_x, train_y, epochs=1, batch_size=64)
    print(' current type of sequential model after fit %s ' % type(model_se))
    model_se_res = model_se.predict(test_x)
    print(' current type of sequential model predict res type %s and value %s ' % (type(model_se_res), model_se_res))
    # 即便是使用sequential 构建的model 也可以当做函数使用
    # 即当需要预测结果的时候可以使用model.predict 或者是model(X)
    # 但是后者要求输入类型为tensor类型
    # model_se(train_x)


    # 另外一个问题 sequential 训练好的模型当需要灵活重用的时候 需要使用函数式模型接口
    # 以上面的单隐层softmax 分类网络为例 获取隐藏层输出 以及从隐藏层输入内容获取最终的输出结果
    # 首先获取隐藏层的结果
    print('-'*30+' how to use the functional api to rebuild a flexible model '+'-'*30)
    # 获取层数 可以传递name 以及 index 0 表示输入层
    # 层类型为预定义类型 库中定义类型
    print(model_se.get_layer(index=0))
    # 输出类型为tensor类型
    model_se_hiddenout = model_se.get_layer(index=1).output
    print(' current hidden layer out of the sequential type % s' % type(model_se_hiddenout))
    # 以隐藏层的输出 作为输入 而 最终输出层输出作为输出构建一个新的modle
    # 直接使用出问题 要求输入的类型为inputLayer 类型 所以
    # 若果想从已经训练好的模型中提取部分作为新的模型 如果是从中间隐藏层截取
    # 那么输入层不能够直接使用前层的输出 必须构建一个inputlayer 型的tensor类型作为input参数
    new_input = Input(shape=(64,))
    print(' type new input %s ' % type(new_input))
    # model_se_new = Model(input=model_se_hiddenout, output=model_se.get_layer(index=2).output)
    # model_se_new = Model(input=new_input, output=model_se.get_layer(index=2).output)
    # 上述方式依然不能够解决问题 因为会出现链接不上的问题
    # 所以使用训练的模型中的部分只能够从输入到某一个隐藏层构建新的模型
    # 不能够从某一个隐藏层到最终输出层构建模型 因为这样会出现截断的情况


    # 小结 sequence 和 functional 构建模型的两种方式
    # functional 构建模型的方式更加的底层 输入输出涉及的类型基本都是底层的框架的张量类型(tf中的tensor)
    # sequence 构建方式更加顶层 不涉及张量的相关技术 这种模型是functional 构建的特殊模型
    # 当需要使用到fit好的模型的中间结果的时候 可以采用functional构建一个新的模型
    # 输入使用fit好的模型的输入(张量类型) 输出使用某一个中间层的输出(张量类型)
    # 无论是functional 或者是sequence 构建的模型(以及各层)都可以用函数的方式调用 但操作的数据类型为张量类型
    # 输出的类型也为张量类型
