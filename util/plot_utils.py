#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   plot_utils.py
@Time    :   2025/12/26 14:27:35
@Author  :   木白 
@description   :   plot the pic according to the data.
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def set_figsize(figsize = (3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    set the axes of matplotlib.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,xscale="linear", yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(6, 4), axes=None):
    """
    draw the data points.
    """
    if legend is None:
        legend = []
    set_figsize(figsize)
    # get current axes.
    axes = axes if axes else plt.gca()
    
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    # clear the axes.
    axes.cla()
    
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
def save_plot(file_path, save_dir="images"):
    cwd_path = os.getcwd()
    rel_path = os.path.relpath(file_path, cwd_path)
    pic_name = "_".join(rel_path.split('/'))+".png"
    save_dir = os.path.join(cwd_path, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, pic_name)
    try:
        plt.savefig(save_path)
    except:
        assert "Failed to save " + save_path
    
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
        Plot a list of images.
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # image tensor
            ax.imshow(img.numpy())
        else:
            # PIL image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes            
    
    
    
if __name__ == "__main__":
    x = np.arange(0, 3, 0.1)
    plot(x, [3*x**2-4*x,2*x-3], 'x', 'y', legend=['sin(x)', '2*x-1'])
    plt.show()
    
    
    
        
        
    