#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   initializer.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/18 下午5:08   aleaho      1.0        

"""
文档说明：


"""

# import lib

import torch.nn as nn
import numpy as np


def initial_transitions(transitions):
    scope = np.sqrt(1 / transitions.size(0))
    nn.init.uniform_(transitions, a=-scope, b=scope)



