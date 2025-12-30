#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fashion_mnist.py
@Time    :   2025/12/28 11:13:19
@Author  :   sss
@description   :   FashionMNIST
'''
import datasets.transforms as T
FashionMNIST = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
def make_fashion_mnist_transforms(image_set):
    return T.ToTensor()

def get_fashion_mnist_labels(labels):
    """
    return the text label of fashion mnist.
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


