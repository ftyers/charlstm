# https://github.com/infiniteoverflow/Char-RNN/blob/master/Char-RNN.ipynb

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os

from charnn import CharLSTM, one_hot_encode

if(torch.cuda.is_available()):
    print("GPU is present")
    gpu = True
else:
    print("No GPU Available")
    gpu = False


def predict(net,char,h=None,top_k=None):
    
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x,len(net.chars))
    inputs = torch.from_numpy(x)
    
    if(gpu):
        inputs = inputs.cuda()
        
    h = tuple([each.data for each in h])
    out,h = net(inputs,h)
    
    p = F.softmax(out,dim=1).data
    
    if(gpu):
        p = p.cpu()
        
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p,top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
    
    p = p.numpy().squeeze()
    
    char = np.random.choice(top_ch,p=p/p.sum())
    
    return net.int2char[char],h

def sample(net, size, prime='The', top_k=None):
        
    if(gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

if __name__=="__main__":
    
    # Here we have loaded in a model that trained over 20 epochs `rnn_20_epoch.net`
    with open('rnn_20_epoch.net', 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
        
    loaded = CharLSTM(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])
    
    # Sample using a loaded model
    print(sample(loaded, 200, top_k=5, prime="The "))
    
    
