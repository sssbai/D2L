#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/12/30 22:23:41
@Author  :   sss
@description   :   the concise implementation of linear regression+soft_max
'''
import torch
class LinRegSm(torch.nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.block = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.num_inputs, self.num_classes),
        )
    def forward(self, x):
        x = self.block(x)
        return x
    
def cross_entropy_loss(logits, labels):
    results = torch.nn.functional.cross_entropy(logits, labels)
    return results
    

class SetCriterion(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
    
    def forward(self, logits, labels):
        return cross_entropy_loss(logits, labels)
    
class PostProcess(torch.nn.Module):
        
    def forward(self, logits:torch.Tensor, labels:torch.Tensor):
        if logits.ndim > 1 and logits.size(-1) > 1:
            preds = logits.argmax(dim=-1)
        else:
            preds = logits
            
        correct_mask = preds == labels
        return {
            "preds": preds,
            "labels":labels, 
            "num_correct": correct_mask.sum(),
            "num_samples": labels.numel()
        }
        