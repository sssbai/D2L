#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2025/12/29 16:14:47
@Author  :   sss
@description   :   xxxxxxxxx
'''
import torch

def sgd(params:torch.Tensor, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
class SGD():
    def __init__(self, params, lr:float=1e-3):
        self.params = params
        print(id(self.params), id(params))
        self.lr = lr
        
    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad
                
    def zero_grad(self):
        # print(self.params)
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
        
        
    