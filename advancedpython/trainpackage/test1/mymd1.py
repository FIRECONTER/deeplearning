"""Module1."""
import os

class Test1(object):
    """This is test class one."""
    def __init__(self, name):
        self.name = name
        print('the path is: '+__file__)
        print('real path is '+os.path.realpath(__file__))
        # 使用__file__永远获取当前文件的绝对路径 无论执行上下文在哪里
