#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   engine.py
@Time    :   2025/12/30 22:23:12
@Author  :   sss
@description   :   xxxxxxxxx
'''


import torch
from util.misc import Accumulator
import util.misc as util
from datasets.fashion_mnist_eval import FashionMnistEvaluator
def train_one_epoch(model:torch.nn.Module, criterion, data_loader, optimizer, epoch):
    if isinstance(model, torch.nn.Module):
        model.train()
        
    metric_logger = util.MetricLogger(delimiter="\t")

    header = "Epoch: [{}]".format(epoch)
    print_freq = 600
    for x,y in metric_logger.log_every(data_loader, print_freq, header, show_log=False):
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(loss=loss.item())
        
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, args):
    if isinstance(model, torch.nn.Module):
        model.eval()
    metric_logger = util.MetricLogger()
    header = "Test:"
    print_freq = 100
    
    fashionmnist_evaluator = FashionMnistEvaluator(args)
    for x, y in metric_logger.log_every(data_loader, print_freq, header, show_log=False):
        y_hat = model(x)
        results = postprocessors(y_hat, y)
        metric_logger.update(test_accuracy=results['num_correct']/results['num_samples'])
        fashionmnist_evaluator.update(results)
    
        
    print("Test stats:",metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats, fashionmnist_evaluator
        
    