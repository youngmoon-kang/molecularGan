# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:00:51 2020

@author: SFC202004009
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from utils2 import *
import sys
from autoencoder_model import get_model
import random

load_model = True

def make_partition():
    smi_list = read_smiles('ZINC.smiles', 50000)
    test_list = smi_list[45000:]
    len_total = [len(x) for x in smi_list]
    
    num_train = 45000
    num_test = 500

    smi_list = smiles_to_number(smi_list)
    smi_list = np.array(smi_list)
    len_total = np.array(len_total)
    
    smi_train = smi_list[0:num_train]
    smi_test = smi_list[(num_train) : ]
    len_train = len_total[0:num_train]
    len_test = len_total[(num_train) : ]
    
    train_dataset = DataSet(smi_train, len_train)
    test_dataset = DataSet(smi_test, len_test)
    
    partition = {'train': train_dataset,
                 'test': test_dataset}
    
    print('train start!')
    return partition, test_list

def sort_array(x, lengths, labels):
    lengths, sorted_idx = lengths.sort(0, descending=True)
    x = x[sorted_idx]
    
    return x, lengths

class DataSet(Dataset):
    def __init__(self, smi_list, len_list):
        self.len_list = len_list
        self.smi_list = smi_list
        
    def __len__(self):
        return len(self.len_list)
    
    def __getitem__(self, index):
        return self.smi_list[index], self.len_list[index]

def train(encoder, decoder, partition, enc_optimizer, dec_optimizer, criterion):
    batch_size = 512
    encoder.train()
    decoder.train()
    
    dataloader = DataLoader(partition['train'], batch_size = batch_size, shuffle = False, num_workers = 0)
    
    total_loss = 0
    for ep, data in enumerate(dataloader):
        x, lengths = data
        x = x.float()
        bs = x.shape[0]
        x = x.cuda()
        lengths = lengths.cuda()
        
        lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]
        # label = label[sorted_idx]
        
        hidden = encoder(x, lengths)
        prev_output = np.zeros((bs, 1, 256), dtype = np.long)
        prev_output = torch.from_numpy(prev_output).long().cuda()
        loss = 0

        for i in range(120):            
            prev_output, hidden, output = decoder(prev_output, hidden)
            output = output.squeeze()
            target = x[:, i]
            loss += criterion(output, target.long())
        
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        
        loss.backward()
        total_loss += loss.item()
        
        enc_optimizer.step()
        dec_optimizer.step()
        print('\r{} : {}'.format(ep, len(dataloader)), end = '')
        
    total_loss /= len(dataloader)
    return encoder, decoder, total_loss

def test(encoder, decoder, s):
    smi = s
    lengths = np.array([len(s[0])])
    
    smi_num = np.array(smiles_to_number(smi))
    x = torch.from_numpy(smi_num).float().cuda()
    
    hidden = encoder(x, lengths)
    prev_output = np.zeros((1, 1, 256), dtype = np.long)
    prev_output = torch.from_numpy(prev_output).long().cuda()
    result = np.zeros((1, 120))
    
    for i in range(120):
        prev_output, hidden, output = decoder(prev_output, hidden)
        _topv, topi = output.topk(1)
        result[0, i] = topi
            
    print(result)
    print('target: ', smi[0])
    print('predicted: ', num_to_smiles(result))

partition, test_list = make_partition()

def load_model_func():
    encoder, decoder = get_model()
    
    encoder.load_state_dict( torch.load('./save/encoder299.pth'))
    encoder.eval()
    
    decoder.load_state_dict( torch.load('./save/decoder299.pth'))
    decoder.eval()
    
    encoder.cuda()
    decoder.cuda()
    
    rand_num = random.randint(0, 300)
    test(encoder, decoder, test_list[rand_num : rand_num + 1])
    
    return encoder, decoder

encoder, decoder = get_model()

if(load_model):
    encoder, decoder = load_model_func()

# enc_optimizer = optim.Adam(encoder.parameters(), lr = 0.0005, weight_decay = 1e-4)
# dec_optimizer = optim.Adam(decoder.parameters(), lr = 0.0005, weight_decay = 1e-4)
enc_optimizer = optim.SGD(encoder.parameters(), lr = 0.00003, weight_decay = 1e-4)
dec_optimizer = optim.SGD(decoder.parameters(), lr = 0.00003, weight_decay = 1e-4)
criterion = nn.CrossEntropyLoss()

for i in range(240, 300):
    encoder.cuda()
    decoder.cuda()
    
    # lr = 0.0005 * (0.99 ** (i))
    lr = 0.00001 * (0.99 ** (i))
    enc_optimizer = optim.SGD(encoder.parameters(), lr =lr, weight_decay = 1e-4)
    dec_optimizer = optim.SGD(decoder.parameters(), lr = lr, weight_decay = 1e-4)
    
    tic = time.time()
    encoder, decoder, _loss = train(encoder, decoder, partition, enc_optimizer, dec_optimizer, criterion)
    tok = time.time()
    print('\repoch: {} loss: {:.2f}'.format(i, _loss), 'took: {:.2f}'.format( tok - tic))
    
    if(i % 10 == 9):
        torch.save(encoder.state_dict(), './save/encoder{}.pth'.format(i))
        torch.save(decoder.state_dict(), './save/decoder{}.pth'.format(i))
        
        rand_num = random.randint(0, 300)
        test(encoder, decoder, test_list[rand_num : rand_num + 1])
