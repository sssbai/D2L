#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2025/12/29 16:37:29
@Author  :   sss
@description   :   xxxxxxxxx
'''

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())