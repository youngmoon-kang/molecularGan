# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:47:03 2020

@author: SFC202004009
"""


import torch
import torch.nn as nn
import numpy as np

a = np.array([1, 5])
a = torch.from_numpy(a).long()

emnbedding = nn.Embedding(31, 100, padding_idx = 0)

result = emnbedding(a)
print(result.shape)
print(result)
