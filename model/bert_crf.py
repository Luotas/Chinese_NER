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
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from model.crf import CRF1

# torch.log_softmax()

# def log_softmax(self: Tensor, dim: _int, dtype: Optional[_dtype]=None)
# def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
log_soft = F.log_softmax


class Bert_CRF(BertPreTrainedModel):

    def __init__(self, config):
        super(Bert_CRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

        self.crf = CRF1(self.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids, )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        attn_mask = attention_mask.type(torch.uint8)

        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_mask, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_mask)
            return prediction
