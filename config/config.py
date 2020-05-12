#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   config.py    
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/13 下午8:56   aleaho      1.0        

"""
文档说明：


"""

# import lib

from configparser import ConfigParser


class myconf(ConfigParser):

    def __init__(self, defaults=None):
        super(myconf, self).__init__(defaults=defaults)
        self.sec = "Additional"

    def optionxform(self, optionstr):
        return optionstr


class Configruable(myconf):
    def __init__(self, config_file):
        super(Configruable, self).__init__()

        config = myconf()
        config.read(filenames=config_file)
        self.config_file = config_file
        self._config = config

        print('Load config successfully.')
        for section in self._config.sections():
            for k, v in self._config.items(section):
                print(k, ":", v)

    def add_args(self, key, value):
        self._config.set(self.sec, key, value)
        self._config.write(open(self.config_file, 'w'))

    # Data
    @property
    def train_dir(self):
        return self._config.get('Data', 'train_dir')

    @property
    def dev_dir(self):
        return self._config.get('Data', 'dev_dir')

    @property
    def test_dir(self):
        return self._config.get('Data', 'test_dir')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    # [Save]
    @property
    def save_pkl(self):
        return self._config.getboolean('Save', 'save_pkl')

    @property
    def pkl_directory(self):
        return self._config.get('Save', 'pkl_directory')

    @property
    def pkl_data(self):
        return self._config.get('Save', 'pkl_data')

    @property
    def save_dict(self):
        return self._config.getboolean('Save', 'save_dict')

    @property
    def dict_directory(self):
        return self._config.get('Save', 'dict_directory')

    @property
    def word_dict(self):
        return self._config.get('Save', 'word_dict')

    @property
    def label_dict(self):
        return self._config.get('Save', 'label_dict')

    @property
    def save_model(self):
        return self._config.getboolean('Save', 'save_model')

    @property
    def save_best_model_dir(self):
        return self._config.get('Save', 'save_best_model_dir')

    @property
    def model_name(self):
        return self._config.get('Save', 'model_name')

    @property
    def apr_dir(self):
        return self._config.get('Save', 'apr_dir')

    # Model
    @property
    def use_crf(self):
        return self._config.getboolean('Model', 'use_crf')
    @property
    def average_batch(self):
        return self._config.getboolean('Model', 'average_batch')

    @property
    def gradient_acc_steps(self):
        return self._config.getint('Model', 'gradient_acc_steps')

    # Optimizer

    @property
    def crf_learning_rate(self):
        return self._config.getfloat('Optimizer', 'crf_learning_rate')


    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def eps(self):
        return self._config.getfloat('Optimizer', 'eps')

    @property
    def use_lr_decay(self):
        return self._config.getboolean('Optimizer', 'use_lr_decay')

    @property
    def lr_rate_decay(self):
        return self._config.getfloat('Optimizer', 'lr_rate_decay')

    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizier', 'min_lrate')

    # Train

    @property
    def epochs(self):
        return self._config.getint('Train', 'epochs')

    @property
    def batch_size(self):
        return self._config.getint('Train', 'batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Train', 'test_batch_size')

    @property
    def backward_batch_size(self):
        return self._config.getint('Train', 'backward_batch_size')

    @property
    def log_interval(self):
        return self._config.getint('Train', 'log_interval')
