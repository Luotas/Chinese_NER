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
from common.main_help import *
from trainer import Train


def start_train(trainloader, testloader, model, config):
    T = Train(trainloader=trainloader, testloader=testloader, model=model, config=config)
    T.train()


def main(config):
    trainloader, testloader = load_data(config)

    model = load_model(config)
    print(model)

    start_train(trainloader, testloader, model, config)


# def deal_with_data(config):
#     trainloader, testloader = load_data(config)
#     for x, y in trainloader:
#         max_size, y = prepare_data(y, config.tag2idx, config)    #
#
#     a, b, c = convert_examples_to_features(x[:1], None, config.tokenizer, seq_length=max_size)
#
#     print(x[0])
#     print(len(list(x[0].strip().split(" "))))
#     print(a)
#     print(b)
# #     print(c)
# #     print('*****************')
# #     print(y.shape)
#     print(y)
# #     mask = torch.ne(y, config.padID)
# #     print(max_size)
# #     print('*****************')
# #     print(mask)
# #     print(mask.long())
#     break


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



    # model = load_model(config)
    # print(model)
