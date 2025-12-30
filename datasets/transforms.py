#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   transforms.py
@Time    :   2025/12/29 10:12:24
@Author  :   sss
@description   :  Transforms and data argumentation for image
'''

import torchvision.transforms as T
import torchvision.transforms.functional as F

class ToTensor():
    # args 是为了接收位置参数，kwds 是为了接收关键字参数
    # def __call__(self, *args, **kwds):
    def __call__(self, img):
        return F.to_tensor(img)