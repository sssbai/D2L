#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/12/30 22:23:41
@Author  :   sss
@description   :   the concise implementation of linear regression+soft_max
'''
import torch
class LinRegSm(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.block = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.num_inputs, self.num_outputs),
        )
    def forward(self, x):
        x = self.block(x)
        return x
        