# -*- coding:utf-8 -*-
"""Learning to use package and module in python."""


import os
# from test1 import mymd1,mymd2 # 可以from型导入方式也可以按照下面的方式导入
# from test1.mymd1 import *
# from test1.mymd1 import Test1
# import test1.mymd1  # 这种导入方式也是对的
# import test1.mymd1  # import XX.XX.XX 的方式只能够到module不能够到class
# fromt test1 import *
# 当没有init 内容时词句出错，test1只是一个目录只能够通过import test1.module 访问或者 from test1.module import 因为没有init的文件夹只能够当路径用不能够当module用
from test1 import Test1
if __name__ == '__main__':
    # print(type(test1))  # 导入的文件夹无论里面有什么都会被当做module 即便为空
    # print(Test1.mymd1)  # 输出错误说module 没有mymd1 和mymd2 因为init里面什么也没有
    # print(test1.mymd2)
    # print(mymd1) # 打印module 显示详细信息 其实module 全名为test1.mymd1
    # print(type(mymd1))  # 类型为module
    # print(test1.mymd1) # 要正确的输出 必须使用 import test1.mymd1 from import 导入法会使得输出有问题
    # print(Test1) # import * 方式可以完全导入模块中内容，但是编码不推荐这种方式 容易覆盖，因此操作的底层基本都是module
    # print(type(__file__))  # 当前文件的路径
    # print(Test1('hello'))
    # print(mymd1)  # 当是由import test1.mymd1 导入时 这种方式出问题不能直接打印只能够 print test1.mymd1
    # print(mymd1)  # 当init 里面使用all 指定全导入模块 那么from test1 import * 中all 所有内容可见
    # print(mymd2)  # all 里面没有mymd2 就无法获取 当all 里面添加了就能够获取了
    # print(mymd1)

    # 更高级的挎包调用 比如sub2 包里面调用 sub1 包里面的内容
