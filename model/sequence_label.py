#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   sequence_label.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/17 10:23   aleaho      1.0        

"""
文档说明：


"""

# import lib

import torch
import torch.nn as nn
from transformers import BertForTokenClassification


class SequenceLabel(nn.Module):

    def __init__(self, config):
        '''
        序列模型初始化，主要包括BertForTokenClassification,CRF
        Args:
            kwargs:

        '''
