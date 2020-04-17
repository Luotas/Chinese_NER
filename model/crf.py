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
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

        device = self.device
        tag_size  = self.tag_size

        # 定义转移矩阵

        #

        print('CRF Model')





