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
# https://www.cnblogs.com/caseast/p/6085837.html


import os
import torch

import config.config as configurable
from data_utils.corpus import PeopelDailyCorpus
from common.utils import download_pretrain_bert_model
from common.main_help import *
from trainer import Train

from data_utils.ner_dataset import NER_Dataset
import numpy as np


def start_train(train_iter, test_iter, model, config):
    T = Train(train_iter=train_iter, test_iter=test_iter, model=model, config=config)
    T.train()


def main(config):
    train_iter, test_iter = load_data(config)

    model = load_model(config)
    print(model)

    start_train(train_iter, test_iter, model, config)


def parse_arguments():
    config_file = './config/config.cfg'
    config = configurable.Configruable(config_file=config_file)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return config


if __name__ == '__main__':
    config = parse_arguments()
    torch.manual_seed(hy.seed_num)

    if config.device != hy.cpu_device:
        torch.cuda.manual_seed(hy.seed_num)

    download_pretrain_bert_model(config)

    main(config)
