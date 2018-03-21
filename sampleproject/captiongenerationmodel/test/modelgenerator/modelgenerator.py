# -*- coding:utf-8 -*-
"""
Description:
    1) 更深层次理解python中的生成器 以及yield关键字
    2) 合理使用model.fit_generator这个方法 的使用
    3) 测试XOR的分类问题
Author； allocator
"""
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


def data_generator(train_x, train_y, batch_size):
    """Generator."""
    # the length of all the trainingSS data set
    length = len(train_x)
    while True:
        for i in range(0, length, batch_size):
            batch_x = []
            batch_y = []
            for j in range(i, min(length, i+batch_size)):
                batch_x.append(train_x[j])
                batch_y.append(train_y)
        # yield a tuple to define each batch of the training data
        yield (np.array(batch_x), np.array(batch_y))


def define_model():
    """Define a simple MLP model."""
    input_1 = Input(shape=(2,), name="input_1")
    hidden_1 = Dense(units=20, activation='sigmoid')(input_1)
    output = Dense(units=1, activation='sigmoid')(hidden_1)
    model = Model(inputs=input_1, outputs=output)
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    print(' current model summary ')
    print(model.summary())
    return model

if __name__ == '__main__':
    train_x = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
    train_y = np.array([0, 0, 1, 1])
    batch_size = 32
    current_model = define_model()
    current_model.fit(train_x, train_y, batch_size=batch_size, epochs=1000, verbose=0)
    pred_y = current_model.predict(train_x)
    print('use normal fit function to train the xor model and result')
    print(pred_y)
    print(type(pred_y))

    # now try to use the fit_generator

    data_length = len(train_x)
    steps = int((data_length/batch_size))
    other_model = define_model()
    other_model.fit_generator(data_generator(train_x, train_y, batch_size), steps_per_epoch=steps, epochs=5000, verbose=0)
    pred_y = other_model.predict(train_x)
    eva_res = other_model.evaluate(train_x, train_y)
    print('use generator to train the xor model and result')
    print(pred_y)
    print(type(pred_y))
    print('eva result')
    print(eva_res)
    print(type(eva_res))


    # fit 和 fit_generator 中各个参数的意义详解
    # 本质上将都是非常重要的fit model 的过程 其中有一些参数的意义是一样的 而有一些参数的意义却不同
    # 两个函数最大的区别在于效率问题 fit_generator 每一次迭代会使用生成器产生的batch data 进行模型的训练
    # 实现生成器生成数据 模型训练两个过程可以并行进行 比如可以使用CPU 进行每个batch size data 的生成 然后在GPU上面完成模型训练
    # 相同的名称相容的意义参数
    # 1) epochs 表示训练结束时的一个整数值 如果initial_epoch 并未指定 那么这个值就是总的迭代次数 如果initial_epoch 指定了
    # 那么真正的迭代次数为 epochs-initial_epoch
    # 2) verbose verbosity mode 表示日志显示 取值为0(不显示日志) 1(只是显示进度) 2(每个epoch 迭代过程都会打印日志)
    # 3) callbacks 回调函数列表 参数类型为list 在training 过程中的某一个时间会调用这里面的回调函数
    # 4) initial_epoch 指定训练的起始整数值 这个值的意义在于有可能在模型训练过程中断了 同样的模型可能需要继续训练
    # 此时可以设置这个值 从上次结束训练的整数值开始继续训练 知道表征训练次数的整数值等于epochs
    # 5) shuffle 一般是布尔值,表示每一次迭代前 对于这次迭代的一个batch_size的数据是否进行洗牌(顺序的随机打乱) 默认是True
    # 6) steps_per_epoch
    # 在fit 训练函数中
    # 一次迭代过程中的步数 一般即是data_size/batch_size 根据mini gd 过程 一次epoch 会使用所有的training data
    # 一个batch_size 的数据会用来更新权重 而dataset 中有多个大小的batch_size数据 因此这个值也表示一个epoch 需要进行
    # 权重更新的次数
    # 而在生成器版本中 这个参数是一个重要的参数 表示当生成器返回这个参数个 大小的数据时 表示一个epoch 结束进而进行下一个epoch
    # 生成器每次返回的数据是一个batch_size 的数据 所以本质上说 这个参数也表示一次epoch 需要进行权重更新的次数即是batch 的数量
    # 在普通fit函数以及生成器函数中这个值意义是一样的
    # 相同的参数名称意义可能有差别
    # 2) validation_data
    # 3) validation_steps
    # 4) class_weight
    # 完全不相同的参数
    # 1) validation_split(fit)
    # 2) sample_weight(fit)
    # 3) max_queue_size(fit_generator)
    # 默认为10
    # 4) workers(fit_generator)
    # 默认为1 表示训练中使用的进程数量
    # 5) use_multiprocessing(fit_generator)
    # 6) fit 训练数据参数 x 全部的训练样本的list 可以是numpy.ndarray 类型数据 , y 全部训练样本的labels numpy.ndarray类型
    # 虽然描述可以是list 中存放ndarray 类型作为输入数据 但是一般都完全转换为ndarray 类型数据.
    # 7) fit_generator 训练数据参数 为一个生成器 生成器有要求
    # 第一: 要求生成器每一次产生一个batch_size 的数据 所以batch_size 这个参数值是要传递进生成器的 这也可以解释
    # 为什么在生成器版本的训练函数(fit_generator)中没有batch_size 这个参数 因为这个参数已经传递到生成器中
    # 第二: 生成器返回值类型要求, 是一个tuple 型数据(list数据也可以, 可以进行自动转换)
    # 返回要求的tuple 格式为
    # (inputs, targets) 或者是(inputs, targets, sample_weights) 类型的数据 外层的这个tuple 可以换成list或者ndarray 类型数据
    # 其中inputs 和 targets 包含的样本数量是一样的. 如果需要训练的数据有多个输入 那么inputs 也是一个tuple(list) 里面按照顺序放置多输入
    # 网络的训练数据
    # 第三: 理论上这个生成器可以无限生成数据 因为epochs不确定 所以要求生成器可以无限生成数据, 所以生成器必须在while 1(True）
    # 逻辑当中, 在所有的Data 通过某种方式遍历完全之后又从第一个数据开始遍历产生用于training 的batch_size 的数据