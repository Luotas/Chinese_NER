#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   sequence_label.py
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/5/17 16:06   aleaho      1.0        

"""
文档说明：


"""

# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bert_crf import Bert_CRF
from model.crf import CRF

log_soft = F.log_softmax


class Sequence_Label(nn.Module):

    def __init__(self, config):

        super(Sequence_Label,self).__init__()

        self.num_labels = len(config.tag2idx)

        self._bert = Bert_CRF.from_pretrained(config.bert_model_dir, num_labels=self.num_labels)

        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                labels=None):
        output = self._bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        attn_mask = attention_mask.type(torch.uint8)

        if labels is not None:
            loss = -self.crf(log_soft(output, 2), labels, mask=attn_mask, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(output, mask=attn_mask)
            return prediction
