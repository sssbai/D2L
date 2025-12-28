#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2025/12/27 22:30:48
@Author  :   sss
@description   :   optimizer
'''
import torch
from torch import nn

def sgd(model:nn.Module, learning_rate):
    return torch.optim.SGD(model.parameters(), lr=learning_rate)
