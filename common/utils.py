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
import sys
import os


def unison_shuffled_copies(x_data, y_data):
    assert len(x_data) == len(y_data), 'data length must equal label length.'

    data = list(zip(x_data, y_data))
    random.shuffle(data)
    x_data, y_data = zip(*data)
    return x_data, y_data



def download_pretrain_bert_model(config):

    # 必须使用该方法下载模型，然后加载
    from flyai.utils import remote_helper
    path = remote_helper.get_remote_date('https://www.flyai.com/m/RoBERTa_zh_L12_PyTorch.zip')
    print(path)

    config.bertConfig_path =os.path.join(sys.path[0], 'data', 'input','model','config.json')

    config.bert_model_dir = os.path.join(sys.path[0], 'data', 'input','model')