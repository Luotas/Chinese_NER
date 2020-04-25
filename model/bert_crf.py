#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   bert_crf.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/25 下午10:56   aleaho      1.0        

"""
文档说明：


"""

# import lib

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from model.crf import CRF


class Bert_CRF(BertPreTrainedModel):

    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None, ):
        pass
