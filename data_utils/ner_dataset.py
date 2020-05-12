#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   ner_dataset.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/5/8 下午8:55   aleaho      1.0        

"""
文档说明：


"""

# import lib

import torch.utils.data as data
from transformers.tokenization_bert import BertTokenizer
import linecache
import common.hyperparameter as hy


class NER_Dataset(data.Dataset):

    def __init__(self, file_path, tag2idx, tokenizer_path='', do_lower_case=True):
        self.tag2idx = tag2idx
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
        self._file_path = file_path

        with open(file_path, 'r', encoding='utf-8') as fp:
            self._lines_count = len(fp.readlines())

    def __len__(self):

        return self._lines_count

    def __getitem__(self, idx):

        line = linecache.getline(self._file_path, idx + 1)
        _sentence, _label = line.strip().split(",")
        _sentence, _label = _sentence.split(), _label.split()

        label = []
        for x in _label:
            if x in self.tag2idx.keys():
                label.append(self.tag2idx[x])
            else:
                label.append(self.tag2idx['O'])
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append(hy.csl)
        # append dummy label 'X' for subtokens
        modified_labels = [self.tag2idx[hy.x]]
        for i, token in enumerate(_sentence):
            if len(bert_tokens) >= 512:
                break
            orig_to_tok_map.append(len(bert_tokens))
            modified_labels.append(label[i])
            new_token = self.tokenizer.tokenize(token)
            bert_tokens.extend(new_token)
            modified_labels.extend([self.tag2idx[hy.x]] * (len(new_token) - 1))

        bert_tokens.append(hy.sep)
        modified_labels.append(self.tag2idx[hy.x])
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        if len(token_ids) > 511:
            token_ids = token_ids[:512]
            modified_labels = modified_labels[:512]
        return token_ids, len(token_ids), orig_to_tok_map, modified_labels, _sentence
