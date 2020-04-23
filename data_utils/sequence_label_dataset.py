#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   sequence_label_dataset.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/23 下午10:39   aleaho      1.0        

"""
文档说明：
Read the csv file into Dataset.
./data/test.csv
./data/train.csv
"""
from torch.utils.data import Dataset
import linecache


class SequenceLabelingDataset(Dataset):
    def __init__(self, filename):
        self._filename = filename
        with open(filename, "r", encoding="utf-8") as f:
            self._lines_count = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        return line.strip().split(",")

    def __len__(self):
        return self._lines_count
