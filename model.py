# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:18:36 2020

@author: SFC202004009
"""

import torch
import torch.nn as nn
import numpy as np
import torch.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.l1 = nn.Linear(100, 128)
        
        self.l2 = nn.Linear(128, 128)
        
        self.l3 = nn.Linear(128, 256)
        
        self.drop_out = nn.Dropout(0.8)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu( self.l1(x) )
        x = self.drop_out(x)
        
        x = self.relu( ( self.l2(x) ) )
        x = self.drop_out(x)
        
        x = self.tanh( ( self.l3(x) ) )
        
        return x
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(256, 128)
        
        self.l2 = nn.Linear(128, 64)
        
        self.l3 = nn.Linear(64, 32)
        
        self.l4 = nn.Linear(32, 1)
        
        self.drop_out = nn.Dropout(0.8)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu( ( self.l1(x) ) )
        x = self.drop_out(x)
        
        x = self.relu( ( self.l2(x) ) )
        x = self.drop_out(x)
        
        x = self.tanh( ( self.l3(x) ) )
        x = self.drop_out(x)
        
        x = self.l4(x)
        
        return x
    
def get_model_gen_dis():
    g = Generator()
    d = Discriminator()
    return g, d

if ( __name__ == "__main__" ):
    x = np.random.randint(0, 31, (5, 20))
    x = torch.from_numpy(x)
    
    lengths = [20] * 5
    lengths = np.array(lengths)