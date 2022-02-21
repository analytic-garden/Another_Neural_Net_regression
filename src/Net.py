#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Net.py - a simple neural net class for regression
author: Bill Thompson
license: GPL 3
copyright: 2022-02-18
"""

import torch

# a sumple NN class
class Net(torch.nn.Module):    
    def __init__(self, layer1_out = 200, layer2_out = 100):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(1, layer1_out)
        self.linear2 = torch.nn.Linear(layer1_out, layer2_out)
        self.linear3 = torch.nn.Linear(layer2_out, 1)

        self.leaky_rlu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_rlu(x)
        x = self.linear2(x)
        x = self.leaky_rlu(x)
        x = self.linear3(x)
        return x
