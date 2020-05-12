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
import torch
from tqdm import tqdm, trange
from common.utils import prepare_data
import timeit
import datetime
import subprocess
import common.hyperparameter  as hy

import os
import shutil

from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class Train:

    def __init__(self, **kwargs):
        '''

        Args:
            kwargs:
                train_iter : train data iter
                train_iter  : test data iter
                model       : nn model
                config      : config

        '''

        print('Training Starting.....')

        self.train_iter = kwargs['train_iter']
        self.test_iter = kwargs['test_iter']
        self.model = kwargs['model']
        self.config = kwargs['config']
        self.use_crf = self.config.use_crf
        self.average_batch = self.config.average_batch
        self.t_total = len(self.train_iter) // self.config.gradient_acc_steps * self.config.epochs
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

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

    def _get_optimizer(self):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW( optimizer_grouped_parameters,lr=self.config.learning_rate, eps=self.config.eps)

        # optimizer = AdamW([{'params': self.model.bert.parameters()},
        #                    {'params': self.model.classifier.parameters()},
        #                    {'params': self.model.crf.parameters(), 'lr': self.config.crf_learning_rate}],
        #                   lr=self.config.learning_rate, eps=self.config.eps)

        return optimizer

    def _get_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.t_total)

        return scheduler

    def train(self):

        unique_labels = list(self.config.tag2idx.keys())

        device = self.config.device
        global_step = 0

        for param in list(self.model.parameters())[:-21]:
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

        self.model.zero_grad()
        self.model.train()
        training_loss = []
        validation_loss = []
        train_iterator = trange(self.config.epochs, desc="Epoch", disable=0)
        start_time = timeit.default_timer()

        for epoch in (train_iterator):
            epoch_iterator = tqdm(self.train_iter, desc="Iteration", disable=-1)
            tr_loss = 0.0
            tmp_loss = 0.0
            self.model.train()
            for step, batch in enumerate(epoch_iterator):
                s = timeit.default_timer()
                token_ids, attn_mask, _, labels, _, _ = batch
                # print(labels)
                inputs = {'input_ids': token_ids.to(device),
                          'attention_mask': attn_mask.to(device),
                          'token_type_ids': None,
                          'labels': labels.to(device)
                          }

                loss = self.model(**inputs)
                loss.backward()
                loss.detach()
                tmp_loss += loss.item()
                tr_loss += loss.item()
                if (step + 1) % 1 == 0:
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                if step == 0:
                    print('\n%s Step: %d of %d Loss: %f' % (
                        str(datetime.datetime.now()), (step + 1), len(epoch_iterator), loss.item()))
                if (step + 1) % self.config.log_interval == 0:
                    print('%s Step: %d of %d Loss: %f' % (
                        str(datetime.datetime.now()), (step + 1), len(epoch_iterator), tmp_loss / 1000))
                    tmp_loss = 0.0

            print("Training Loss: %f for epoch %d" % (tr_loss / len(self.train_iter), epoch))
            training_loss.append(tr_loss / len(self.train_iter))

            # '''
            # Y_pred = []
            # Y_true = []

            val_loss = 0.0
            self.model.eval()

            if not os.path.isdir(self.config.apr_dir):
                os.mkdir(self.config.apr_dir)

            predict_file = os.path.join(self.config.apr_dir, 'prediction_' + str(epoch) + '.csv')

            writer = open(predict_file, 'w', encoding='utf-8')
            for i, batch in enumerate(self.test_iter):
                token_ids, attn_mask, org_tok_map, labels, original_token, sorted_idx = batch
                # attn_mask.dt
                inputs = {'input_ids': token_ids.to(device),
                          'attention_mask': attn_mask.to(device),
                          'token_type_ids': None,
                          }

                dev_inputs = {'input_ids': token_ids.to(device),
                              'attention_mask': attn_mask.to(device),
                              'token_type_ids': None,
                              'labels': labels.to(device)
                              }
                with torch.torch.no_grad():
                    tag_seqs = self.model(**inputs)
                    tmp_eval_loss = self.model(**dev_inputs)
                val_loss += tmp_eval_loss.item()
                # print(labels.numpy())
                y_true = list(labels.cpu().numpy())
                for i in range(len(sorted_idx)):
                    o2m = org_tok_map[i]
                    pos = sorted_idx.index(i)
                    for j, orig_tok_idx in enumerate(o2m):
                        writer.write(original_token[i][j] + '\t')
                        writer.write(unique_labels[y_true[pos][orig_tok_idx]] + '\t')
                        pred_tag = unique_labels[tag_seqs[pos][orig_tok_idx]]
                        if pred_tag == hy.x:
                            pred_tag = 'O'
                        writer.write(pred_tag + '\n')
                    writer.write('\n')

            validation_loss.append(val_loss / len(self.test_iter))
            writer.flush()
            print('Epoch: ', epoch)
            command = "python conlleval.py < " + predict_file
            process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            result = process.communicate()[0].decode("utf-8")
            print(result)

            writer = open(os.path.join(self.config.apr_dir, 'result' + str(epoch) + '.text'), 'w', encoding='utf-8')
            writer.write(result)
            writer.flush()

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': tr_loss / len(self.train_iter),
            }, self.config.apr_dir + 'model_' + str(epoch) + '.pt')

        total_time = timeit.default_timer() - start_time
        print('Total training time: ', total_time)
        return training_loss, validation_loss
