#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   engine.py
@Time    :   2025/12/29 16:13:59
@Author  :   sss
@description   :   xxxxxxxxx
'''

import torch
from util.misc import Accumulator
import util.misc as util
from soft_max.manual_implementation.evaluation import accuracy
def train_one_epoch(model:torch.nn.Module, criterion, data_loader, optimizer, epoch):
    if isinstance(model, torch.nn.Module):
        model.train()
        
    metric_logger = util.MetricLogger(delimiter="\t")
    # metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('class_error', util.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('grad_norm', util.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 600
    metric = Accumulator(3)
    # for x, y, in data_loader:
    for x,y in metric_logger.log_every(data_loader, print_freq, header):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        # grad_norm = 
        optimizer.step()
        metric.add(float(loss)*len(y), accuracy(y_hat, y), y.size().numel())
        metric_logger.update(loss=loss.item())
        metric_logger.update(train_accuracy=accuracy(y_hat, y)/y.size().numel())
        
    print("Averaged stats:", metric_logger)
    # return metric[0]/metric[2], metric[1]/metric[2]
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader):
    print(id(data_loader))
    if isinstance(model, torch.nn.Module):
        model.eval()
    metric_logger = util.MetricLogger()
    header = "Test:"
    print_freq = 100
    for x, y in metric_logger.log_every(data_loader, print_freq, header):
        y_hat = model(x)
        results = postprocessors(y_hat, y)/y.size().numel()
        metric_logger.update(test_accuracy=results)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
    