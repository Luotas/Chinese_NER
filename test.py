#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   test.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/5/8 下午9:04   aleaho      1.0        

"""
文档说明：


"""

# import lib

# from collections import OrderedDict
#
# label_set = ['B_ORG','I_ORG','B_ORG','I_ORG','B_PER']
#
# print(label_set)
# d = list(OrderedDict.fromkeys(label_set))
#
# print(d)


import torch

A = torch.tensor([1,3,2])

print(A)

print(A.cuda())




