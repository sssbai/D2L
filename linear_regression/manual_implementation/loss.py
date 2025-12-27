#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2025/12/26 16:24:44
@Author  :   sss
@description   :   the manual implementaion of loss.
'''

import torch

def squared_loss(y_hat, y):
    """
        The squared loss.
    """
    return (y_hat-y.reshape(y_hat.shape))**2/2
