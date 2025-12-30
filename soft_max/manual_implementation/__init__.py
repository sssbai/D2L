#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2025/12/30 13:32:09
@Author  :   sss
@description   :   xxxxxxxxx
'''

from .model import build_linreg_sm

def build_model(args):
    return build_linreg_sm(args)