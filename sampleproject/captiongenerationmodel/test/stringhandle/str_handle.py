#-*- coding:utf-8 -*-
"""
Description: str.maketrans and translate
"""
import string


if __name__ == '__main__':
    # maketrans 为内建的函数,用于构建两个相同长度的映射表(前参数为被转换字符串,后一参数为转换之后的字符串) 返回数据类型为dict
    # 转换之后的table
    l1 = 'this1'
    l2 = 'awvt2'
    res = str.maketrans(l1, l2)
    print(res)
    print(type(res))
    t1 = 'ha'
    print(t1.translate(res))
    # maketrans 一般和translate联合使用
    # maketrans 有第三个参数 用作转换, 第三个参数也是字符串,当进行转换时,这个字符串中字符会被translate 成None
    # 一个非常经典的应用 将字符串中的标点符号全部去掉
    res_1 = str.maketrans('', '', string.punctuation)
    t2 = 'dsff,errr,,sdsd:dsdsd?'
    print(t2.translate(res_1))  # t2中所有的标点符号可以被去除 标点符号一般在word划分的时候是跟随word在一起的
    # 将调用translate 的字符串中的字符根据trans table 中的映射关系进行转换