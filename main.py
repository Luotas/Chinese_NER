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

from data_utils.ner_dataset import  NER_Dataset
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


def pad(batch):
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i>0) for i in ids] for ids in tok_ids]
    LT = torch.LongTensor
    label = do_pad(3, maxlen)

    # sort the index, attn mask and labels on token length
    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    tok_ids = LT(tok_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]
    labels = LT(label)[sorted_idx]
    org_tok_map = get_element(2)
    sents = get_element(-1)

    return tok_ids, attn_mask, org_tok_map, labels, sents, list(sorted_idx.cpu().numpy())



def generate_train_data(config,tag2idx):

    import torch.utils.data as data
    from tqdm import tqdm

    train_dataset = NER_Dataset(file_path = config.train_dir,
                              tag2idx = tag2idx,
                              tokenizer_path = config.bert_vocab_path)


    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)



    epoch_iterator = tqdm(train_iter, desc="Iteration", disable=-1)

    for step, batch in enumerate(epoch_iterator):
        tok_ids, attn_mask, org_tok_map, labels, sents,sorted_idx= batch
        print(tok_ids)



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


    token2idx, idx2token, tag2idx, idx2tag = dump_tags(config)


    generate_train_data(config,tag2idx)



    # model = load_model(config)
    # print(model)
