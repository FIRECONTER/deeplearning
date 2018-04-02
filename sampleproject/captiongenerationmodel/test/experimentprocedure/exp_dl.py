# -*- coding:utf-8 -*-
"""
Descriptions:
    1) how to experiment on one dl model
    2) A vs A test: 因为模型存在一定的随机性 所以这个测试在实际的调参之前 保证模型在其他环境 或者多次运行 能够得到相当的结果
        (比如相似的均值 方差等指标) 如果几次实验结果相差较大 可以适当的提升模型repeat的次数 保证最大化的减小随机因子对模型的
        影响
Author: allocator
Time: 26/03/2018
"""