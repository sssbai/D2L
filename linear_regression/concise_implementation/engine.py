#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   engine.py
@Time    :   2025/12/27 22:32:37
@Author  :   sss
@description   :   engine
'''
from torch import nn
from typing import Iterable
def train_one_epoch(model:nn.Module, criterion:nn.Module, data_loader:Iterable, optimizer:nn.Module):
    metrics = {"train_loss":[]}
    for x, y in data_loader:
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        metrics["train_loss"].append(loss.item())
        optimizer.step()
    return metrics
        
    
