#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2026/01/07 15:51:31
@Author  :   sss
@description   :   the concise implementation of mlp
'''
import torch
class MLP(torch.nn.Module):
    def __init__(self,num_inputs, num_classes, num_mid=256):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(num_inputs, num_mid)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_mid, num_classes)
        
        # init the parameters of mlp
        torch.nn.init.normal_(self.linear1.weight, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, std=0.01)
        
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
def cross_entropy_loss(logits, labels):
    result = torch.nn.functional.cross_entropy(input=logits, target=labels)
    return result
    
class SetCriterion(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
    def forward(self, logits, labels):
        return cross_entropy_loss(logits, labels)

class PostProcess(torch.nn.Module):
    def forward(self, logits, labels):
        if logits.ndim > 1 and logits.size(-1) > 1:
            preds = torch.argmax(logits, dim=-1)
        else:
            preds = logits
        correct_mask = preds == labels
        return {
            "preds":preds,
            "labels":labels,
            "num_correct": correct_mask.sum(),
            "num_samples":labels.numel(),            
        }
    
        
