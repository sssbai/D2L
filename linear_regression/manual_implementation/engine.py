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
def train_one_epoch(model, loss, batch_size, features, labels, optimizer, epochs:int, lr = 0.03,w_size=(3,1)):
    w = torch.normal(0, 0.01, w_size, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    metrics = {"val_loss":[], "train_loss":[]}
    for epoch in range(epochs):
        for x, y in data_iter(batch_size, features,labels):
            y_hat = model(x, w, b)
            l = loss(y_hat, y)
            l.sum().backward()
            metrics["train_loss"].append(l.sum().item())
            optimizer([w, b], lr, batch_size)
        with torch.no_grad():
            l = loss(model(features, w, b), labels)
            metrics["val_loss"].append(l.mean().item())
            print(f"epoch {epoch + 1}, loss {float(l.mean()):f}")
    return metrics, {"w":w.detach(), "b":b.detach()}
    
    
            
            
