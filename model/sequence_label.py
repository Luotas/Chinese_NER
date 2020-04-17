#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   sequence_label.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/17 10:23   aleaho      1.0        

"""
文档说明：


"""

# import lib

import torch
from torch import Tensor
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from model.crf import CRF


class SequenceLabel(nn.Module):

    def __init__(self, config):
        '''
        序列模型初始化，主要包括BertForTokenClassification,CRF
        Args:
            kwargs:

        '''
        super(SequenceLabel,self).__init__()

        # Bert
        bert_config_path = config.bertConfig_path
        bert_model_dir = config.bert_model_dir

        # device
        device = config.device

        # label_num
        num_labels = config.tag_size + 2 if config.use_crf else config.tag_size

        # model

        self.bertConfig = BertConfig.from_json_file(bert_config_path)
        self.bertConfig.num_labels = num_labels
        self.encoder = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=bert_model_dir,
                                                                     config=self.bertConfig)

        if config.use_crf:
            args = {'device': device, 'tag_size': num_labels}
            self.crf_layers = CRF(**args)



    def forward(self, input: Tensor):
        pass

