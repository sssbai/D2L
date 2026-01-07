#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/12/28 09:48:46
@Author  :   sss
@description   :   
'''

import torch

def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp/partition

# def linreg_sm(x, w, b):
#     return softmax(torch.matmul(x.reshape((-1, w.shape[0])),w)+b)

class LinRegSm(torch.nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.w = torch.normal(0, 0.01, size=(num_inputs, num_classes), requires_grad=True)
        self.b = torch.zeros(num_classes, requires_grad=True)
    
    def parameters(self):
        yield self.w
        yield self.b
        
    def forward(self, x):
        return softmax(torch.matmul(x.reshape((-1, self.w.shape[0])),self.w)+self.b)


def build_linreg_sm(args):
    return LinRegSm(
        num_inputs = args.num_inputs,
        num_classes = args.num_classes
        )
