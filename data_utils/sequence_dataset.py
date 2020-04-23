#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   sequence_dataset.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/23 下午9:52   aleaho      1.0        

"""
文档说明：


"""

# import lib
from typing import List

from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, x_data: List[List[str]], labels: List[List[str]]):
        '''
        Make the word/label pair with Dataset . It's easy to separate in to batch by DataLoader .
        Args:
            x_data (List[List[str]]) : word of sentence
            labels (List[List[str]]) : label of word
        '''
        super(SequenceDataset, self).__init__()
        assert len(x_data) == len(labels), ' sentence size must equal label size.'
        self.x_data = x_data
        self.labels = labels
        self._count = len(x_data)

    def __getitem__(self, index):
        return (' '.join(self.x_data[index]), ' '.join(self.labels[index]))

    def __len__(self):
        return self._count
