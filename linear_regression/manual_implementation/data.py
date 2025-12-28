#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2025/12/26 14:10:33
@Author  :   木白 
@description   :   generate the synthetic data.
'''

import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

def synthetic_data(w, b, num_examples):
    """
        generate the data of y = w * x + b + bias.
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y = y + torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

if __name__ == "__main__":
    # python linear_regression/manual_implementation/data.py
    w = torch.tensor([2.1, -3.4], dtype=torch.float32)
    x, y = synthetic_data(w, 4.2, 1000)
    print(x.shape, y.shape)
    from util.plot_utils import *
    plt.scatter(x[:, 1].detach().numpy(), y.detach().numpy(), 10)
    save_plot(os.path.abspath(__file__))
    plt.show()