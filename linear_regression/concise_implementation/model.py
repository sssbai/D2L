#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/12/27 22:28:05
@Author  :   sss
@description   :   the concise implementation of model.
'''

import torch
from torch import nn

def linreg(input_d, output_d):
    model = nn.Sequential(nn.Linear(input_d, output_d))
    model[0].weight.data.normal_(0, 0.01)
    model[0].bias.data.normal_(0, 0.01)
    return model