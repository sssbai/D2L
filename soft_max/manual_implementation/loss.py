#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2025/12/29 16:15:32
@Author  :   sss
@description   :   xxxxxxxxx
'''
import torch

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

class SetCriterion(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        pass
    def forward(self, outputs, targets):
        num = len(targets)
        loss = -torch.log(outputs[range(len(targets)), targets]).sum()/len(targets)
        return loss