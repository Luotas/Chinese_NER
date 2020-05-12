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

    # main(config)

    # token2idx, idx2token, tag2idx, idx2tag = dump_tags(config)
    #
    # print(tag2idx)
    #
    # generate_train_data(config, tag2idx)

    # model = load_model(config)
    # print(model)

    # epoch_iterator = tqdm(data_iter, desc="Iteration", disable=-1)
    #
    # for step, batch in enumerate(epoch_iterator):
    #     tok_ids, attn_mask, org_tok_map, labels, sents, sorted_idx = batch
    #     print(tok_ids)

    from tqdm import tqdm

    train_iter, test_iter = load_data(config)

    epoch_iterator = tqdm(train_iter, desc="Iteration", disable=-1)

    i = 0
    for step, batch in enumerate(epoch_iterator):
        tok_ids, attn_mask, org_tok_map, labels, sents, sorted_idx = batch

        print(tok_ids.shape)
        print(f'{tok_ids[0]}\n{tok_ids[1]}')
        print(f'{attn_mask[0]}\n{attn_mask[1]}')
        print(f'{org_tok_map[0]}\n{org_tok_map[1]}')
        print(f'{labels[0]}\n{labels[1]}')
        print(f'{sents[0]}\n{sents[1]}')
        print(f'{len(sents[0])},{len(sents[1])}')
        # print(sorted_idx)
        i +=1
        if i == 10:
            break
