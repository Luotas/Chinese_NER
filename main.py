#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   main.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/13 下午8:15   aleaho      1.0        

"""
文档说明：


"""

# import lib

import os
import torch

import config.config as configurable
from data_utils.corpus import PeopelDailyCorpus
from common.utils import download_pretrain_bert_model
from common.main_help import load_model


def parse_arguments():
    config_file = './config/config.cfg'
    config = configurable.Configruable(config_file=config_file)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return config


if __name__ == '__main__':
    config = parse_arguments()

    # x_data,y_data = PeopelDailyCorpus.load_corpus(config,'train')
    #
    # print(x_data[:5],y_data[:5])

    download_pretrain_bert_model(config)
    config.tag_size = 5
    model = load_model(config)
    print(model)
