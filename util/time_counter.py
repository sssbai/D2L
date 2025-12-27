#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   time_counter.py
@Time    :   2025/12/26 15:06:21
@Author  :   木白 
@description   :   record the time of running.
'''
import time
import numpy as np

class Timer:
    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        self.tik = time.time()
        
    def stop(self):
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times)/len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
if __name__ == "__main__":
    import torch
    c = torch.zeros(10)
    timer = Timer()
    for i in range(10):
        c[i] = i+1
    print(f"The time is {timer.stop():.5f} sec.")
        
