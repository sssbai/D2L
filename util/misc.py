#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   misc.py
@Time    :   2025/12/30 10:32:29
@Author  :   sss
@description   :   miscellaneous utilities
'''
import time
import torch
import datetime
from collections import defaultdict, deque

class Accumulator:
    """
    the accumulation of n metrics
    """
    def __init__(self, n):
        self.data = [0.0]*n
    
    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0]*len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total/self.count
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median = self.median,
            avg=self.avg,
            global_avg = self.global_avg,
            max=self.max,
            value=self.value
        )
        
        
class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwards):
        for k, v in kwards.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
            
    def __getattr__(self, attr):
        """
        This function is used to apply the following code:
        `
            a = MetricLogger()
            a.loss # a.loss == a.meters["loss"]
        `
        """
        
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        # type(self).__name__ is equal to self.__class__.__name__
        raise AttributeError("'{}' object has no attribute '{}'".fromat(type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
        
    def log_every(self, iterable, print_freq, header=None, show_log=True):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":"+str(len(str(len(iterable))))+"d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0'+space_fmt+'}/{1}]',
                'eta: {eta}',# estimated time of completion.
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}', # max allocated gpu memory. 
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0'+space_fmt+'}/{1}]',
                'eta: {eta}',# estimated time of completion.
                '{meters}',
                'time: {time}',
                'data: {data}',
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time()-end)
            yield obj
            iter_time.update(time.time()-end)
            if show_log:
                if i % print_freq == 0 or i == len(iterable) - 1:
                    eta_seconds = iter_time.global_avg * (len(iterable) - i)
                    # str(datetime.timedelta(seconds=int(eta_seconds))) is equal to datetime.timedelta(seconds=int(eta_seconds)).__str__()
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    if torch.cuda.is_available():
                        print(log_msg.format(
                            i,len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),data=str(data_time),
                            memory=torch.cuda.max_memory_allocated()/MB
                        ))
                    else:
                        print(log_msg.format(
                            i,len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),data=str(data_time),
                        ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        print('{} Total time: {} ({:.4f} s/ it(erable))'.format(
            header, total_time_str, total_time/len(iterable)
        ))
        
        
        
        
            
    
        