#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   trainer.py
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/25 下午3:31   aleaho      1.0        

"""
文档说明：


"""

# import lib

from common.utils import prepare_data
from common.extract_features import convert_examples_to_features
import torch.optim as optim
import torch.nn as nn
import torch

import time


class Train:

    def __init__(self, **kwargs):
        '''

        Args:
            kwargs:
                trainloader : train data loader
                testloader  : test data loader
                model       : nn model
                config      : config

        '''

        print('Training Starting.....')

        self.trainloader = kwargs['trainloader']
        self.testloader = kwargs['testloader']
        self.model = kwargs['model']
        self.config = kwargs['config']
        self.use_crf = self.config.use_crf
        self.average_batch = self.config.average_batch
        self.optimizer = optim.Adam([{'params': self.model.encoder.parameters(), 'lr': self.config.bert_learning_rate},
                                     {'params': self.model.crf_layer.parameters()}],
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay, eps=self.config.eps)
        self.loss_function = self.model.crf_layer.neg_log_likelihood_loss

    @staticmethod
    def _get_model_args(data, tags, config):
        max_size, tags = prepare_data(tags, config.tag2idx, config)

        input_ids, input_mask, input_type_ids = convert_examples_to_features(data=data, config=config,
                                                                             seq_length=max_size)

        mask = torch.ne(tags, config.padID).to(config.device)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=config.device)
        input_mask = torch.tensor(input_mask, dtype=torch.long, device=config.device)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long, device=config.device)

        return input_ids, input_mask, input_type_ids, tags, mask

    def _optimizer_batch_step(self, config, backward_count):
        """
        :param config:
        :param backward_count:
        :return:
        """
        if backward_count % config.backward_batch_size == 0 or backward_count == self.config.batch_size:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _calculate_loss(self, feats, mask, tags):
        loss_value = self.loss_function(feats, mask, tags)

        if self.average_batch:
            batch_size = feats.size(0)
            loss_value /= float(batch_size)

        return loss_value

    def train(self):
        epochs = self.config.epochs
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            backward_count = 0

            for batch_data, tags in self.trainloader:
                backward_count += 1

                input_ids, input_mask, type_ids, tags, mask = self._get_model_args(batch_data, tags, self.config)

                (logit,) = self.model(input_ids=input_ids, token_type_ids=type_ids, position_ids=input_mask)
                loss = self._calculate_loss(feats=logit, mask=mask, tags=tags)
                loss.backward()

                self._optimizer_batch_step(self.config, backward_count)

                print(f'loss:{loss.detach().item()}')
