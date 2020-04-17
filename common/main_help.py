#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   main_help.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/17 16:36   aleaho      1.0        

"""
文档说明：


"""

# import lib

from model.sequence_label import SequenceLabel
from common.hyperparameter import cpu_device


def load_model(config):

    model = SequenceLabel(config)

    if config.device!=cpu_device:
        model = model.to(device=config.device)

    return model






