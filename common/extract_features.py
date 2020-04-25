#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @File    :   extract_features.py
# @Contact :   aleaho@live.com
# @License :   

# @Modify Time      @Author    @Version    
# ------------      -------    --------    
# 2020/4/24 10:11   aleaho      1.0        

"""
文档说明：


"""


# import lib

def convert_examples_to_features(data, config, seq_length=100):
    '''
    convert train dataset to features.
    Args:
        data:
        config:
        seq_length:

    Returns:

    '''
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for text in data:
        input_ids, input_mask, input_type_ids = convert_example_to_feature(text_a=text, text_b=None,
                                                                           tokenizer=config.tokenizer,
                                                                           seq_length=seq_length)

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_input_type_ids.append(input_type_ids)

    return all_input_ids, all_input_mask, all_input_type_ids


def convert_example_to_feature(text_a, text_b, tokenizer, seq_length=110):
    '''
    对文本text_a, text_b以字为单位进行分隔（Bert是以字为单位）
    根据最大序列长度的值，对文本进行合适的截断
    对于单句，需要在头尾分别加上[CLS]和[SEP]的标志位，因此单句的长度若大于最大序列长度 -2则进行截断，截断原则是从句子开头到最大长度。
    对于两个句子，需要在头尾中分别加上[CLS][SEP][SEP]三个标志位，因此判断len(a)+len(b)是否大于max_seq_length-3，
    若长度大于max_seq_length-3，则从尾部截断长度较长的句子单字。

    对tokens_a和tokens_b进行处理，增加标志位，构建其他几个变量，规则如下

    对于成对序列
    tokens: [CLS] 你 好 吗 ? [SEP] 我 是 中 国 人 。 [SEP]
    type_ids: 0 0 0 0 0 0 1 1 1 1 1 1 1
    对于单句序列:
    tokens: [CLS] 你 好 吗 ?[SEP]
    type_ids: 0 0 0 0 0 0
    tokens = []存放处理后的序列
    input_type_ids[]存放每个句子对应的标签（句子1的序列都为0，句子2的序列都为1）
    input_ids存放处理后的序列对应的id值（根据vocab.txt）
    input_mask 用来标注实际值掩膜（即有值的位标1，否则标0）

    处理后长度不足max_seq_length的补零

    Args:
        text_a:
        tonkenizer:
        max_len:

    Returns:
            input_ids (list): 存放处理后的序列对应的id值
            type_ids (list): 存放每个句子对应的标签
            input_mask (list):用来标注实际值掩膜（即有值的位标1，否则标0）
    '''

    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None

    if text_b:
        # tokens_b = tokenizer.tokenize(text_b)
        tokens_b = list(text_b)

    # if tokens_b:
    #     # Modifies `tokens_a` and `tokens_b` in place so that the total
    #     # length is less than the specified length.
    #     # Account for [CLS], [SEP], [SEP] with "- 3"
    #     # 对于两个句子，需要加上头尾，中间，一共加上三个标注位
    #     # _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    #     pass
    #
    # else:
    #     if len(tokens_a) > seq_length - 2:
    #         tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append('[SEP]')
    input_type_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append('[SEP]')
        input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.  只关注有值存在的地方
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return input_ids, input_mask, input_type_ids
