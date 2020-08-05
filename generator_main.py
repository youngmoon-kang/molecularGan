# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:13:29 2020

@author: SFC202004009
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from utils2 import *
import sys
from autoencoder_model import get_model
from model import get_model_gen_dis


pre_train = True

class DataSet(Dataset):
    def __init__(self, smi_list, len_list):
        self.len_list = len_list
        self.smi_list = smi_list
        
    def __len__(self):
        return len(self.len_list)
    
    def __getitem__(self, index):
        return self.smi_list[index], self.len_list[index]

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

def make_noise(batch_size):
    noise = np.random.rand(batch_size, 100)
    return noise

def train(encoder, gen_model, dis_model, dataset, gen_optimizer, dis_optimizer, gen_criterion, dis_criterion):
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = True, num_workers = 0)
    
    gen_model.train()
    dis_model.train()
    encoder.train()
    gen_total_loss = 0
    dis_total_loss = 0
    for i, data in enumerate(dataloader):
        x, lengths = data
        bs = len(x)
        x = x.cuda()
        lengths = lengths.cuda()
        lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]
        real_data = encoder(x, lengths)
        
        noise = make_noise(bs)
        noise = torch.from_numpy(noise).float().cuda()
        fake_data = gen_model(noise)
        
        dis_output = dis_model(real_data)
        
        zero_label = np.zeros(shape = (bs,1))
        zero_label = torch.from_numpy(zero_label).cuda()
        one_label = np.ones(shape = (bs,1))
        one_label = torch.from_numpy(one_label).cuda()
        
        """train discriminator"""
        dis_output_cri = dis_output.view(bs, 1)
        dis_optimizer.zero_grad()
        dis_loss_real = dis_criterion(dis_output_cri, one_label)
        dis_loss_fake = dis_criterion(dis_model(fake_data), zero_label)
        dis_loss = dis_loss_real + dis_loss_fake
        dis_total_loss += dis_loss.item()
        
        dis_loss.backward(retain_graph=True)
        dis_optimizer.step()
        
        """train generator"""
        gen_optimizer.zero_grad()
        
        gen_loss = gen_criterion(dis_model(fake_data), one_label)
        gen_total_loss += gen_loss.item()
        
        gen_loss.backward()
        gen_optimizer.step()
        print('\r{} / {}'.format(i, len(dataloader)), end='')
        
    return gen_model, dis_model, dis_total_loss/len(dataloader), gen_total_loss/len(dataloader)

def test(gen, decoder):
    gen.eval()
    decoder.eval()
    sample = make_noise(1)
    sample = torch.from_numpy(sample).float()
    sample = sample.cuda()
    hidden = gen(sample)
    hidden = hidden.unsqueeze(0)
    
    prev_output = np.zeros((1, 1, 256), dtype = np.long)
    prev_output = torch.from_numpy(prev_output).long().cuda()
    result = np.zeros((1, 120))
    
    for i in range(120):
        prev_output, hidden, output = decoder(prev_output, hidden)
        _topv, topi = output.topk(1)
        result[0, i] = topi
            
    print(result)
    print('predicted: ', num_to_smiles(result))
    # result = num_to_smiles(result)
    # result = np.array([result])
    # print(result.shape)
    # draw_2D(result)
    return

def exprience():
    #dataset, _ = make_partition()
    gen, dis = get_model_gen_dis()
    encoder, decoder = get_model()

    encoder.load_state_dict(torch.load('./save/encoder299.pth'))
    encoder.eval()
    decoder.load_state_dict(torch.load('./save/decoder299.pth'))
    decoder.eval()
    
    if(pre_train):
        gen.load_state_dict(torch.load('./save/gen.pth'))
        gen.eval()
        dis.load_state_dict(torch.load('./save/dis.pth'))
        dis.eval()
    
    encoder.cuda()
    decoder.cuda()
    gen.cuda()
    dis.cuda()
    test(gen, decoder)
    sys.exit()
    
    gen_optimizer = optim.SGD(gen.parameters(), lr = 0.001, weight_decay = 1e-4)
    dis_optimizer = optim.SGD(dis.parameters(), lr = 0.001, weight_decay = 1e-4)
    
    gen_criterion = nn.BCEWithLogitsLoss()
    dis_criterion = nn.BCEWithLogitsLoss()
    
    for i in range(50):
        tik = time.time()
        gen, dis, dis_loss, gen_loss = train(encoder, gen, dis, dataset['train'], gen_optimizer, dis_optimizer, gen_criterion, dis_criterion)
        torch.save(gen.state_dict(), './save/gen.pth')
        torch.save(dis.state_dict(), './save/dis.pth')
        tok = time.time()
        print('\repoch: {} gen_loss: {:.5f} dis_loss: {:.5f} took: {:.2f}'.format(i, gen_loss, dis_loss, tok - tik))
        print()
        
    test(gen, decoder)
    
exprience()
