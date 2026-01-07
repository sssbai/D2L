#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fashion_mnist_eval.py
@Time    :   2025/12/31 10:06:37
@Author  :   sss
@description   :   apply the evalutation of fashion_mnist dataset.
'''
import torch
class FashionMnistEvaluator:
    def __init__(self, args):
        self.num_correct: int = 0
        self.num_samples: int = 0
        self.accuracy: int = 0
        self.num_classes :int = args.num_classes
        self.pred_counts = [0] * self.num_classes
        self.label_counts = [0] * self.num_classes
    def update(self, res:dict):
        self.num_correct += res.get("num_correct", 0)
        self.num_samples += res.get("num_samples", 0)
        if "preds" in res:
            for p in res["preds"].flatten():
                self.pred_counts[p.item()] += 1 
        if "labels" in res:
            for p in res["labels"].flatten():
                self.label_counts[p.item()] += 1 
        
        
    def accumulate(self):
        if self.num_samples > 0:
            self.accuracy = self.num_correct/self.num_samples
    
    def summarize(self, fmt="{:.6f}"):
        print("The accuracy is ", fmt.format(self.accuracy))
    
        
