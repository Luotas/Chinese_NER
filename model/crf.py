#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   crf.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/17 11:42   aleaho      1.0        

"""
文档说明：

REFERENCE : https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py


"""

# import lib


import torch
import torch.nn as nn

from model.initializer import *


def log_sum_exp(vec, m_size):
    '''

    Args:
        vec(Tensor):size=(batch_size,vanishing_dim,hidden_dim)
        m_size:hidden_dim

    Returns:
        
    '''

    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, idx.view(-1, 1, m_size)).view(-1, 1, m_size)

    return max_score.view(-1, m_size) + torch.log(
        torch.sum(
            torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):

    def __init__(self, **kwargs):
        '''

        Args:
            kwargs:
            target_size (int) :target size
            device (torch.device) :device type
        '''
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        device = self.device

        self.START_TAG, self.STOP_TAG = -2, -1

        init_transitions = torch.zeros(self.target_size + 2, self.target_size + 2, device=device)

        initial_transitions(init_transitions)

        init_transitions[:, self.START_TAG] = -10000
        init_transitions[self.STOP_TAG, :] = -10000

        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask):
        '''
        Do the forward algorithm to compute the partition function(batched).
        Args:
            feats (Tensor) :size = (batch_size,seq_len,target_size)
            mask (Tensor) :size= (batch_size,seq_len)

        Returns:

        '''

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        mask = mask.transpose(1, 0).contiguous()

        ins_num = seq_len * batch_size

        '''注意view操作 是.view(ins_num,1,tag_size),不是.view(ins_num,tag_size,1)'''
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)

        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)

        _, inivalues = next(seq_iter)

        '''只取出START_TAG到其他标签的分数'''

        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size, 1)  # batch_size*to_target

        '''
        add start score(from start to all tag,duplicate to batch_size)
        partition = partition + self.transition[START_TAG,:].view(1,tag_size,1).expand(batch_size,tag_size,1)
        iter over last score
        '''

        for idx, cur_values in seq_iter:
            '''
            previous to_target is current from_target
            partition:previous results log(exp(from_target)), # (batch_size*from_target)
            cur_values:batch_size*from_target*to_target
            '''

            cur_values = cur_values + partition.cuntiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            '''replace the partition where the maskvalue=1,other partition value keeps the same'''
            masked_cur_partition = cur_partition.masked_select(mask_idx)

        '''
        until the last state,add transition score for all partition(and do log_sum_exp)
        then select the value in STOP_TAG
        '''

        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + \
                     partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

        cur_partition = log_sum_exp(cur_values, tag_size)

        final_partition = cur_partition[:, self.STOP_TAG]
        return final_partition.sum(), scores

    def _score_sentence(self, scores, mask, tags):
        '''

        Args:
            scores: size = (seq_len,batch_size,tag_size,tag_size)
            mask:   size = (batch_size,seq_len)
            tags:   size = (batch_size,seq_len)

        Returns:
            score
        '''
        # print(scores.size())
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        tags = tags.view(batch_size, seq_len)

        '''  convert tag value into a new format,recorded label bigram information to index '''
        new_tags = torch.empty(batch_size, seq_len, device=self.device, requires_grad=True).log()

        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        ''' transition for label to STOP_TAG '''
        end_transition = self.transitions[:, self.STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ''' Length for batch ,last word position=length - 1 '''
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).log()

        ''' index the label id of last word '''
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ''' index the transition score for end_id to STOP_TAG '''
        end_energy = torch.gather(end_transition, 1, end_ids)

        ''' convert tag as (seq_len, batch_size, 1) '''
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        ''' need convert tags id to search from 400 positions of scores '''
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        ''' 
        add all score togather
        gold_socre = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        '''

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        '''

        Args:
            feats:  size = (batch_size,seq_len,tag_size)
            mask:   size = (batch_size,seq_len)
            tags:   size = (batch_size,seq_len)

        Returns:

        '''

        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats=feats, mask=mask)
        gold_score = self._score_sentence(scores=scores, mask=mask, tags=tags)

        return forward_score - gold_score


    def _viterbi_decode(self,feats,mask):
        '''

        Args:
            feats:  size = (batch_size,seq_len,self.tag_size + 2)
            mask:   size = (batch_size,seq_len)

        Returns:
            decode_idx:(batch_size,seq_len) decoded sequence
            path_score:(batch_size,1) corresponding score for each sequence
        '''
        pass



    def forward(self, feats, mask):
        '''

        Args:
            scores: size = (seq_len,batch_size,tag_size,tag_size)
            mask:   size = (batch_size,seq_len)

        Returns:

        '''

        path_score,best_path = self._viterbi_decoder(feats,mask)

