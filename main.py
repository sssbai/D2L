#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025/12/26 16:26:43
@Author  :   sss
@description   :   the main file of d2l.
'''

import torch
from torch import nn
import matplotlib.pyplot as plt
from util.plot_utils import plot, save_plot

def main_manual_linreg():
    from linear_regression.datasets.data import synthetic_data
    from linear_regression.datasets.data_util import data_iter
    from linear_regression.manual_implementation.model import linreg
    from linear_regression.manual_implementation.loss import squared_loss
    from linear_regression.manual_implementation.optimizer import sgd
    from linear_regression.manual_implementation.engine import train_one_epoch
    
    model = linreg
    criterion = squared_loss
    optimizer = sgd
    learning_rate = 0.03
    batch_size = 10
    epochs=10
    
    w = torch.tensor([2, -3.5, 9])
    b = torch.tensor(-0.4)
    num = 100
    features, labels = synthetic_data(w, b, num)
    plt.scatter(features[:, -1].detach().numpy(), labels.detach().numpy())
    save_plot("linear_regression_data")
    print(f"feature shape is {list(features.shape)}")
    print(f"label shape is {list(labels.shape)}")
    
    metrics = {"val_loss":[],"train_loss":[]}
    init_w = torch.normal(0, 0.01, (3, 1), requires_grad=True)
    init_b = torch.zeros(1, requires_grad=True)
    params = {"w":init_w, "b":init_b}
    for epoch in range(epochs):
        train_data_loader = data_iter(batch_size, features, labels)
        train_stats, params = train_one_epoch(model, criterion, train_data_loader, optimizer, lr=learning_rate, parameter=params, batch_size=batch_size)
        metrics["train_loss"]+=train_stats["train_loss"]
        with torch.no_grad():
            loss = criterion(model(features, params['w'].detach(), params['b'].detach()), labels)
            metrics["val_loss"].append(loss.mean().item())
            print(f"epoch {epoch + 1}, loss {float(loss.mean().item()):f}")
    plot(metrics["train_loss"], xlabel="iter", ylabel="loss", fmts="m--")
    save_plot("manual_linreg_train_loss")
    plot(metrics["val_loss"], xlabel="epoch", ylabel="loss", fmts="g-")
    save_plot("manual_linreg_val_loss")
    pred_w, pred_b = params['w'].detach(), params["b"].detach()
    x = torch.arange(-10, 10, 0.1).reshape((-1, 1))

    w = w.reshape((1, -1))
    pred_w = pred_w.reshape((1, -1))

    y = (w*x + b).sum(-1)
    pred_y = (pred_w*x + pred_b).sum(-1)
    
    plot(x.flatten(), [y, pred_y], xlabel="x", ylabel="y")
    save_plot("manual_linreg_true_vs_pred_epoch_"+str(epochs))
def main_concise_linreg():
    from linear_regression.datasets.data import synthetic_data
    from linear_regression.datasets.data_util import data_iter
    from linear_regression.concise_implementation.model import linreg
    from linear_regression.concise_implementation.optimizer import sgd
    from linear_regression.concise_implementation.engine import train_one_epoch
    epochs = 10
    batch_size=10
    input_dimension = 3
    output_dimension = 1
    model = linreg(input_dimension, output_dimension)
    criterion = nn.MSELoss()
    optimizer = sgd(model, learning_rate=0.03)
    
    w = torch.tensor([2, -3.5, 9])
    b = torch.tensor(-0.4)
    num = 100
    features, labels = synthetic_data(w, b, num)
    plt.scatter(features[:, -1].detach().numpy(), labels.detach().numpy())
    save_plot("linear_regression_data")
    print(f"feature shape is {list(features.shape)}")
    print(f"label shape is {list(labels.shape)}")
    
    metrics = {"val_loss":[], "train_loss":[]}
    for epoch in range(epochs):
        data_loader_train = data_iter(batch_size,features, labels)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer)
        metrics["train_loss"]+=train_stats["train_loss"]
        with torch.no_grad():
            loss = criterion(model(features), labels)
            metrics["val_loss"].append(loss.detach().item())
            print(f"epoch {epoch + 1}, loss {float(loss.detach().item()):f}")
    plot(metrics["train_loss"], xlabel="iter", ylabel="loss", fmts="m--")
    save_plot("concise_linreg_train_loss")
    plot(metrics["val_loss"], xlabel="epoch", ylabel="loss", fmts="g-")
    save_plot("concise_linreg_val_loss")
    pred_w, pred_b = model[0].weight.data, model[0].bias.data
    x = torch.arange(-10, 10, 0.1).reshape((-1, 1))

    w = w.reshape((1, -1))
    pred_w = pred_w.reshape((1, -1))

    y = (w*x + b).sum(-1)
    pred_y = (pred_w*x + pred_b).sum(-1)
    
    plot(x.flatten(), [y, pred_y], xlabel="x", ylabel="y")
    save_plot("concise_linreg_true_vs_pred_epoch_"+str(epochs))
    

    
if __name__ == "__main__":
    main_manual_linreg()
    # main_concise_linreg()
    