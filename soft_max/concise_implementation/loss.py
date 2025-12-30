#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2025/12/30 22:34:17
@Author  :   sss
@description   :   xxxxxxxxx
'''
import torch
def cross_entropy_loss(outputs, targets):
    results = torch.nn.functional.cross_entropy(outputs, targets)
    return results
    

class SetCriterion(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
    
    def forward(self, outputs, targets):
        return cross_entropy_loss(outputs, targets)
        
        
        