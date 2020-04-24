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

import common.hyperparameter as hy

from model.sequence_label import SequenceLabel
from common.hyperparameter import cpu_device

from data_utils.sequence_label_dataset import SequenceLabelingDataset
from torch.utils.data import DataLoader
import transformers.tokenization_bert  as tokenization


def load_model(config):
    model = SequenceLabel(config)

    if config.device != cpu_device:
        model = model.to(device=config.device)

    return model


def load_tokenizer(config):
    tokenizer = tokenization.BertTokenizer(config.bert_vocab_path)

    return tokenizer

def load_data(config):
    if not os.path.isdir(config.pkl_directory): os.makedirs(config.pkl_directory)

    train_dataLoader, test_dataloader = preprocessing(config)
    return train_dataLoader, test_dataloader


def preprocessing(config):

    ''' 1.create tags\index map '''
    token2idx, idx2token, tag2idx, idx2tag = dump_tags(config)
    config.tag2idx = tag2idx
    config.idx2tag = idx2tag
    config.tag_size = len(tag2idx)
    config.padID = tag2idx[hy.pad]

    save_directionary(config, tag2idx, token2idx)

    ''' 2. load bert tokenizer instance '''
    config.tokenizer = load_tokenizer(config)

    ''' 3. create train/test dataset'''

    trainset = SequenceLabelingDataset(config.train_dir)
    testset = SequenceLabelingDataset(config.test_dir)

    ''' 4. initializion the dataloader for loop'''
    train_dataLoader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=config.batch_size, shuffle=True)

    print('******************************')
    print(f'tag_size:{config.tag_size}')
    print(f'idx2tag:{config.idx2tag}')
    print(f'tag2idx:{config.tag2idx}')
    print('******************************')

    return train_dataLoader, test_dataloader


def dump_tags(config):
    token2idx = {hy.pad: 0, hy.unk: 1}
    tag2idx = {hy.pad: 0}
    with open(config.train_dir, "r", encoding="utf-8") as fp:
        for cursor in fp.readlines():
            seq, tags = cursor.split(',')
            for tok in seq.split():
                if tok not in token2idx:
                    token2idx[tok] = len(token2idx)
            for tag in tags.split():
                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)

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
