#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025/12/26 16:26:43
@Author  :   sss
@description   :   the main file of d2l.
'''

import torch
import matplotlib.pyplot as plt
from util.plot_utils import plot, save_plot
from linear_regression.datasets.data import synthetic_data
from linear_regression.manual_implementation.model import linreg
from linear_regression.manual_implementation.loss import squared_loss
from linear_regression.manual_implementation.optimizer import sgd
from linear_regression.manual_implementation.engine import train_one_epoch

def main_linreg():
    learning_rate = 0.03
    batch_size = 10
    epochs=1
    
    w = torch.tensor([2, -3.5, 9])
    b = torch.tensor(-0.4)
    num = 100
    features, labels = synthetic_data(w, b, num)
    plt.scatter(features[:, -1].detach().numpy(), labels.detach().numpy())
    save_plot("linear_regression_data")
    print(f"feature shape is {list(features.shape)}")
    print(f"label shape is {list(labels.shape)}")

    stats, params = train_one_epoch(linreg, squared_loss, batch_size, features, labels, optimizer=sgd, epochs=epochs, lr=learning_rate, w_size=(3,1))
    plot(stats["train_loss"], xlabel="iter", ylabel="loss", fmts="m--")
    save_plot("linreg_train_loss")
    plot(stats["val_loss"], xlabel="epoch", ylabel="loss", fmts="g-")
    save_plot("linreg_val_loss")
    pred_w, pred_b = params['w'], params["b"]
    x = torch.arange(-10, 10, 0.1).reshape((-1, 1))

    w = w.reshape((1, -1))
    pred_w = pred_w.reshape((1, -1))

    y = (w*x + b).sum(-1)
    pred_y = (pred_w*x + pred_b).sum(-1)
    
    plot(x.flatten(), [y, pred_y], xlabel="x", ylabel="y")
    save_plot("linreg_true_vs_pred_epoch_"+str(epochs))
    
    
    
if __name__ == "__main__":
    main_linreg()
    