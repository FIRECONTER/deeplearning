# -*- coding:utf-8 -*-
"""
Description: 研究生成器中添加while 1无限循环的目的 是为了让生成器可以无限的调用 产生无限的循环数据
Author: allocator
"""
def get_data(n):
    while 1:
    # 加上while 1可以让这个生成器产理论上产生无限长数据
    # 如果在for 循环中执行完了继续从初始状态0开始执行这个迭代过程
    # 如果没有while 那么这个生成器是有限长度的 在for 循环次数调用后 后续不会返回结果
        for i in range(n):
            print('current number %d ' % i)
            yield i

if __name__ == '__main__':
    n = 10
    res = []
    k = get_data(10)
    for i in k:
        pass
    print(' the last result ')
    # print(res)