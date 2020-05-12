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

import os
import shutil
from tqdm import tqdm
import numpy as np
import torch

import common.hyperparameter as hy

from common.hyperparameter import cpu_device
from data_utils.ner_dataset import NER_Dataset
import torch.utils.data as data

from model.bert_crf import Bert_CRF


def load_model(config):

    model = Bert_CRF.from_pretrained(config.bert_model_dir,num_labels = len(config.tag2idx))

    if config.device != cpu_device:
        model = model.to(device=config.device)

    return model




def load_data(config):
    if not os.path.isdir(config.pkl_directory): os.makedirs(config.pkl_directory)

    train_iter, test_iter = preprocessing(config)
    return train_iter, test_iter


def preprocessing(config):
    ''' 1.create tags\index map '''
    token2idx, idx2token, tag2idx, idx2tag = dump_tags(config)
    config.tag2idx = tag2idx
    config.idx2tag = idx2tag
    config.tag_size = len(tag2idx)
    config.padID = tag2idx[hy.pad]

    save_directionary(config, tag2idx, token2idx)

    ''' 2. create train/test dataloader'''

    train_iter = generate_data(config,
                               dataset_dir=config.train_dir,
                               tag2idx=tag2idx,
                               batch_szie=config.batch_size)

    test_iter = generate_data(config,
                              dataset_dir=config.test_dir,
                              tag2idx=tag2idx,
                              batch_szie=config.test_batch_size)

    print('******************************')
    print(f'tag_size:{config.tag_size}')
    print(f'idx2tag:{config.idx2tag}')
    print(f'tag2idx:{config.tag2idx}')
    print('******************************')

    return train_iter, test_iter


def pad(batch):
    '''

    Args:
        batch:

    Returns:

    '''
    '''Pads to the longest sample'''
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = get_element(1)
    maxlen = np.array(seq_len).max()
    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    tok_ids = do_pad(0, maxlen)
    attn_mask = [[(i > 0) for i in ids] for ids in tok_ids]
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


def generate_data(config, dataset_dir, tag2idx, batch_szie):
    '''
    生成数据。
    Args:
        config:         配置信息集合
        dataset_dir:    数据集路径
        tag2idx:        标签与编码的对应字典
        batch_szie:

    Returns:

    '''
    dataset = NER_Dataset(file_path=dataset_dir,
                          tag2idx=tag2idx,
                          tokenizer_path=config.bert_vocab_path)

    data_iter = data.DataLoader(dataset=dataset,
                                batch_size=batch_szie,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad)

    return data_iter


def dump_tags(config):
    token2idx = {hy.pad: 0, hy.unk: 1}
    tag2idx = {hy.pad: 0, hy.x: 1}

    with open(config.train_dir, "r", encoding="utf-8") as fp:
        for cursor in fp.readlines():
            seq, tags = cursor.split(',')
            for tok in seq.split():
                if tok not in token2idx:
                    token2idx[tok] = len(token2idx)

            for tag in tags.split():
                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)
                # label_set.append(tag)

    with open(config.test_dir, "r", encoding="utf-8") as fp:
        for cursor in fp.readlines():
            seq, tags = cursor.split(',')
            for tok in seq.split():
                if tok not in token2idx:
                    token2idx[tok] = len(token2idx)
            for tag in tags.split():
                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)

    idx2tag = {}
    for k, v in tag2idx.items():
        idx2tag[v] = k
    idx2token = {}
    for k, v in token2idx.items():
        idx2token[v] = k

    return token2idx, idx2token, tag2idx, idx2tag


def save_directionary(config, label2idx, word2idx):
    '''

    :param config:
    :return:
    '''

    if config.save_dict is True:
        if os.path.exists(config.dict_directory):
            shutil.rmtree(config.dict_directory)
        if not os.path.isdir(config.dict_directory):
            os.makedirs(config.dict_directory)

    config.word_dict_path = '/'.join([config.dict_directory, config.word_dict])
    config.label_dict_path = '/'.join([config.dict_directory, config.label_dict])

    print(f'Word_dict_path:{config.word_dict_path}')
    print(f'Label_dict_path:{config.label_dict_path}')

    save_dict2file(word2idx, config.word_dict_path)
    save_dict2file(label2idx, config.label_dict_path)

    # print(f'copy dictionary to {config.save_dir}')
    # shutil.copytree(config.dict_directory, '/'.join([config.save_dir, config.dict_directory]))


def save_dict2file(dict, path):
    '''

    :param dict:
    :param path:
    :return:
    '''
    print('Saving dictionary!')
    if os.path.exists(path):
        print(f'Path {path} is exist ,deleted.')

    with open(path, 'w', encoding='utf-8') as file:
        for word, index in dict.items():
            file.write(str(word) + '\t' + str(index) + '\n')

    print('Save dictionary finished.')
