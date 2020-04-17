#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   crf.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/17 11:42   aleaho      1.0        

"""
文档说明：


"""

# import lib



import torch
import torch.nn as nn

class CRF(nn.Module):

    def __init__(self,**kwargs):

        super(CRF,self).__init__()
