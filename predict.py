# https://github.com/infiniteoverflow/Char-RNN/blob/master/Char-RNN.ipynb

import sys 
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os

from charlstm import CharLSTM, one_hot_encode

if(torch.cuda.is_available()):
#    print("GPU is present")
    gpu = True
else:
#    print("No GPU Available")
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

    #print('p:', p)
    #print('top_ch:', top_ch)

    res = []
    for sym, p in zip(top_ch, p):
        res.append((net.int2char[sym], sym, p))
    #print('res:', res)
    
    #char = np.random.choice(top_ch,size=3,p=p/p.sum())
    #print(char)
    #topn = [net.int2char[ch] for ch in char]
    #print(topn)
    
    #return net.int2char[char],h
    #return topn,h
    return res,h

def sample(net, size, prime='The', top_k=None):
        
    if(gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
 
    h = net.init_hidden(1)

    hypotheses = {'':1.0}
    hiddens = {'':h}
    
    # First off, run through the prime characters
    prime_chars = [ch for ch in prime] + [None]

    for ch in prime_chars :
        if ch == None: continue
        hypits = [i for i in hypotheses.items()]
        new_hyps = {}
        for hyp, score in hypits:
            char, h = predict(net, ch, h, top_k=top_k)
            new_hyps[hyp+ch] = 1.0
            hiddens[hyp+ch] = h
        hypotheses = new_hyps

#    hypotheses = {''.join(chars):0.0}
#    hypits = [i for i in hypotheses.items()]
#    for hyp, score in hypits:
#        new_hyps = {}
#        for ch in char:
#            new_hyps[hyp+ch] = 0.0
#        hypotheses = new_hyps
           
    print('H:', hypotheses)
    #chars.append(char[0])
    
    generated_chars = 0
    # Now pass in the previous character and get a new one
    while generated_chars < size:
        hypits = [i for i in hypotheses.items()]
        new_hyps = {}
        for hyp, score in hypits:
            #print(hyp, score)
            res, h = predict(net, hyp[-1], hiddens[hyp], top_k=top_k)
            for ch, sym, p in res:
                #print(hyp, '|', ch, '|', sym, '|', p)
                new_hyps[hyp+ch] = hypotheses[hyp] * p
                hiddens[hyp+ch] = h
                generated_chars += 1

        # prune the beam
        hypotheses = {}
        sorted_hyps = [i for i in new_hyps.items()]
        sorted_hyps.sort(key=lambda x: x[1], reverse=True)
        #print('PRUNE', sorted_hyps)
        n_spaces = 0
        for (h,p) in sorted_hyps[:top_k]:
            if h.count(' ') == prime.count(' ') + 1: n_spaces += 1
            hypotheses[h] = p

        if n_spaces == top_k:
            break

        #print('>', hypotheses)
        #chars.append(char)

    #print('H:', hypotheses)

    return hypotheses

if __name__=="__main__":
    
    # Here we have loaded in a model that trained over 20 epochs `rnn_20_epoch.net`
    with open(sys.argv[1], 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
        
    loaded = CharLSTM(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])

    while True:

        prime_ = input('> ')
        
        # Sample using a loaded model
        hyps = sample(loaded, 200, top_k=5, prime=prime_)
        print(' | '.join([h.replace(prime_,'', 1).replace('\n', ' ') for h in hyps]))
#        for h, p in hyps.items():
#            print(p, h)
        
    
