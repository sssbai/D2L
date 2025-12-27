#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2025/12/26 16:30:27
@Author  :   sss
@description   :   the optimizer of model.
'''
import torch

def sgd(params:torch.Tensor, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
    
