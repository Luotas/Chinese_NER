#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   corpus.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/14 下午8:33   aleaho      1.0        

"""
文档说明：


"""

# import lib

from typing import Tuple, List

import data_utils.utils as utils

class DataReader:

    @staticmethod
    def read_corpus(file):
        x_data, y_data = [], []

        with open(file, 'r', encoding='utf-8') as fp:
            x, y = [], []
            for line in fp.readlines():

                if len(line) == 1:
                    x_data.append(x)
                    y_data.append(y)

                    x, y = [], []

                else:
                    words = line.strip().split(' ')
                    x.append(words[0])
                    y.append(words[1])

        return x_data, y_data


class PeopelDailyCorpus():

    @staticmethod
    def load_corpus(config, name):

        file_dir = ''
        if name == 'train':
            file_dir = config.train_dir
        elif name == 'dev':
            file_dir = config.dev_dir
        elif name == 'test':
            file_dir = config.test_dir

        x_data,y_data = DataReader.read_corpus(file_dir)

        if config.shuffle:
            x_data,y_data =utils.unison_shuffled_copies(x_data=x_data,y_data=y_data)




        return  x_data,y_data






