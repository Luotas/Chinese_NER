#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   hyperparameter.py
# @Contact :   aleaho@live.com
# @License :
def load_tokenizer(config):
    tokenizer = tokenization.BertTokenizer(config.bert_vocab_path)

    return tokenizer

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/17 16:38   aleaho      1.0        

"""
文档说明：


"""

# import lib
import torch

seed_num = 237
pad = '[PAD]'   #
unk = '[UNK]'
csl = "[CLS]"
sep = "[SEP]"
cpu_device = torch.device('cpu')
