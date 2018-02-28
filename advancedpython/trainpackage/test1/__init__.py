# -*- coding:utf-8 -*-
"""
关于__init__.py 文件.

这个文件是package必须具有的。如果没有这个package 只是一个层次目录，被外部程序使用时只能够进行最简单的导入操作
比如 import XX.XX.XX from Xx.XX(文件级别) import */Class/function
有了init文件，它可以定义包的一些属性以及方法，让包的用法与文件模块(xxx.py)有一样的功能。
有了init 可以在里面导入包的模块那么在外部调用时只要导入该package 就可以使用package.modulename获得该子文件

"""

# import mymd2  # 貌似不能够这样写了
__all__ = ['mymd1', 'mymd2']
