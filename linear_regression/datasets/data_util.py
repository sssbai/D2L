#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_util.py
@Time    :   2025/12/26 16:08:38
@Author  :   sss 
@description   :   some util of data
'''
import random
import torch

def data_iter(batch_size, features, labels):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_samples)]
        )
        yield features[batch_indices], labels[batch_indices]

