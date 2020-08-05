# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:36:47 2020

@author: SFC202004009
"""


import torch
import torch.nn as nn
import numpy as np

from utils2 import read_smiles, get_logP, smiles_to_number

class Encoder(nn.Module):
    def __init__(self, num_layer = 1, hidden_dim = 256):
        super(Encoder, self).__init__()
        
        self.gru = nn.GRU(100, 256, batch_first = True)
        
        self.embedding = nn.Embedding(31, 100, padding_idx = 0)
        
    def forward(self, x, lengths):
        x = x.long()
        x = self.embedding(x)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.tolist(), batch_first=True)

        packed_output, h = self.gru(packed_input)
        
        return h
    
class Decoder_cell(nn.Module):
    def __init__(self, hidden_size = 256, output_size = 31):
        super(Decoder_cell, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_x, hidden):
        # print(input_x.shape)
        # print(hidden.shape)
        output, hidden = self.gru(input_x.float(), hidden)
        output_save = self.out(output)
        
        return output, hidden, output_save #output_save: 저장용(31개 맞춤), output: 다시 입력용
        
class Decoder(nn.Module):
    def __init__(self, hidden_dim = 256, output_size = 31):
        super(Decoder, self).__init__()
        self.cell = Decoder_cell(hidden_dim, output_size)
        
    def forward(self, hidden): #hidden layer들어감, start도 들어감
        prev_output = np.zeros((hidden.shape[1], 1, 256), dtype = np.long)
        prev_output = torch.from_numpy(prev_output).long()
        
        out = np.zeros((hidden.shape[1], 120, 31))
        for i in range(120):
            prev_output, hidden, output = self.cell(prev_output, hidden)
            output = output.squeeze()
            out[:, i, :] = output.detach().numpy()
            
        return out
    
def get_model():
    g = Encoder()
    d = Decoder_cell()
    
    return g, d

if (__name__ == "__main__"):
    sample = np.random.rand(1, 5, 256)
    sample = torch.from_numpy(sample)
    sample = sample.float()
    d = Decoder()
    result = d(sample)
    print(result.shape)