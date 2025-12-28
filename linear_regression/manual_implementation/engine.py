#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2025/12/26 16:32:41
@Author  :   sss
@description   :   the train file of the manual linear regression.
'''
import torch
from torch import nn
from typing import Iterable
from linear_regression.datasets.data_util import data_iter
from linear_regression.manual_implementation.loss import squared_loss
def train_one_epoch(model, criterion, train_data_loader, optimizer, lr = 0.03, parameter=None, batch_size=1):
    metrics = {"train_loss":[]}
    w = parameter['w']
    b = parameter['b']
    for x, y in train_data_loader:
        y_hat = model(x, w, b)
        l = criterion(y_hat, y)
        l.sum().backward()
        metrics["train_loss"].append(l.sum().item())
        optimizer([w, b], lr, batch_size)
    return metrics, {"w":w, "b":b}
    
    
            
            
