#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/12/26 16:22:35
@Author  :   sss
@description   :   the manual implementation of linear regression.
'''
import torch
from torch import nn

# class LinReg(nn.Module):
#     """
#         The linear regression model.
#     """
#     def __init__(self, input_d, output_d):
#         super().__init__()
#         self.w = torch.normal(0, 0.01, size=(input_d, output_d), requires_grad=True)
#         self.b = torch.zeros(1, requires_grad=True)
    
#     def forward(self, x):
#         return torch.matmul(x, self.w)+self.b
    
def linreg(X, w, b):
    """
        The linear regression model.
    """
    return torch.matmul(X, w) + b

