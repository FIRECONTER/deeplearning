#-*- coding:utf-8 -*-
"""
Use the weights
"""

from keras.models import model_from_json
import numpy as np
import copy


def sigmoid(x):
    """Generate the sigmoid function."""
    return 1.0/(1.0+np.exp(-x))


if __name__ == '__main__':
    train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_y = np.array([0, 1, 1, 0])
    with open('./model_structure.json', 'r') as f:
        model_str = f.read()
    model = model_from_json(model_str)
    model.load_weights('./model_weights.h5')
    # 加载的模型需要重新编译 才可以进行evaluate
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    score = model.evaluate(train_x, train_y)
    print(' current score loss %s and acc %s ' % (score[0], score[1]))
    res = model.predict(train_x)
    print(' the direct predict res is ')
    print(res)

    # direct use the weights
    # model get_config() 可以获取model的配置 后续可以使用Model.from_config(conf) 重新构建model
    # model.layers 属性可以获取整个model的layer 的instance 的 list
    # 首先是model 层的get_layer 以及 get_weights 和 set_weights方法
    # get_layer 以及使用过 根据层名字name 或者index 获取到模型中的层 此用法在用Model重新构建模型时使用过
    # get_weights 方法获取模型中的权重张量 无参数 返回一个list 中间存放numpy array 型数据
    all_weights = model.get_weights()
    print(' current model weights ')
    print(all_weights)
    print(type(all_weights[0]))
    # 关于模型中get_weights 返回的权重list 结构的分析 每一层的W 后面跟上bias 这样的结构 都是ndarray
    # 但是权重相对于常规表示方式有做转置操作

    # 某一层 也可以通过get_weights 获取到权重
    # 有一个问题是如果要获取第一层到第二层的权重那么应该是第二层.get_weights 获取权重
    print('-'*30+' how about the layer weights '+'-'*30)
    hidden_layer = model.get_layer(index=1)
    input_to_hidden_weights = hidden_layer.get_weights()
    print(' current input layer to hidden layer weights is ')
    print(input_to_hidden_weights)
    print(' type of the current input layer weights %s ' % type(input_to_hidden_weights))

    output_layer = model.get_layer(index=2)
    hidden_to_output_weights = output_layer.get_weights()
    print(' current hidden layer to ')
    print(hidden_to_output_weights)

    # 通过权重重新构建整个模型 f(WX+b) 的形式
    Witoh = np.mat(input_to_hidden_weights[0]).T
    Bitoh = np.mat(input_to_hidden_weights[1]).T
    Whtoo = np.mat(hidden_to_output_weights[0]).T
    Bhtoo = np.mat(hidden_to_output_weights[1]).T
    # deep copy the train_x data
    input_data = copy.deepcopy(train_x)

    def model_func(W1, B1, W2, B2, X):
        """Model func with weights and bias."""
        hidden_net = np.array((W1*X+B1).T.tolist()[0])
        # 普通函数不持之直接在list上面计算但是可以在ndarray上面执行计算
        hidden_out = np.mat(sigmoid(hidden_net)).T
        output_net = np.array((W2*hidden_out+B2).T.tolist()[0])
        output_out = sigmoid(output_net)
        return output_out[0]

    # use the weights and bias to predict the input data
    res_weight = []
    for item in train_x:
        X = np.mat(item).T
        tmp_predict = model_func(Witoh, Bitoh, Whtoo, Bhtoo, X)
        res_weight.append(tmp_predict)

    print(' current res from weight is ')
    print(res_weight)
    print(' current res from model is ')
    print(res)
    # 从model.get_weights 获取权重然后根据网络结构计算出来的结果与model.predict结果一样
    # 宗上所述 如果需要使用到模型中的一部分 可以把模型中的权重获取出来 然后使用矩阵运算求得结果
    # 这种方法也是一种可以以某个隐藏层的结果作为输入重新计算输出结果的方式
    # 关键点是获取的权矩阵是ndarray 形式 转换成mat的时候要注意进行转置后计算.
