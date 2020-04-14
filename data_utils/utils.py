#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   utils.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/14 下午9:53   aleaho      1.0        

"""
文档说明：


"""

# import lib

import random

def shuffle(x_data, y_data):
    data = list(zip(x_data, y_data))
    random.shuffle(data)
    x_data, y_data = zip(*data)
    return x_data, y_data
