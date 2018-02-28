# -*- coding:utf-8 -*

import numpy as np
import os

if __name__ == '__main__':
    input_data = '../data/jiayuqingrecent.txt'
    fi = open(input_data, 'r', encoding='utf-8')
    ln_index = 0
    for line in fi.readlines():
        print('current line +%s' % ln_index)
        print(line)
        ln_index = ln_index+1
        if ln_index == 3:
            break
    fi.close()
