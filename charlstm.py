# https://github.com/infiniteoverflow/Char-RNN/blob/master/Char-RNN.ipynb
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def one_hot_encode(arr,n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    one_hot[np.arange(one_hot.shape[0]),arr.flatten()] = 1
    
    one_hot = one_hot.reshape((*arr.shape,n_labels))
    
    return one_hot


def get_batches(arr,batch_size,seq_length):
    batch_ele_count = batch_size * seq_length
    
    batches = len(arr) // batch_ele_count
    
    arr = arr[:(batches*batch_ele_count)]
    
    arr = arr.reshape(batch_size,-1)
    
    for n in range(0,arr.shape[1],seq_length):
        x = arr[:,n:n+seq_length]
        y = np.zeros_like(x)
        
        try:
            y[:,:-1] , y[:,-1] = x[:,1:] , x[:,n+seq_length]
        except IndexError:
            y[:,:-1] , y[:,-1] = x[:,1:] , x[:,0]
            
        yield x,y

if(torch.cuda.is_available()):
#    print("GPU is present")
    gpu = True
else:
#    print("No GPU Available")
    gpu = False

class CharLSTM(nn.Module):
    
    def __init__(self,tokens,n_hidden=256,n_layers=2,drop_prob=0.5,lr=0.001):
        
        super().__init__()
        
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch:ii for ii,ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(self.chars),n_hidden,n_layers,dropout=drop_prob,batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden,len(self.chars))
        
    def forward(self,x,hidden):
        
        r_output , hidden = self.lstm(x,hidden)
        
        out = self.dropout(r_output)
        
        out = out.reshape(-1,self.n_hidden)
        
        out = self.fc(out)
        
        return out,hidden
        
    def init_hidden(self,batch_size):
        
        weight = next(self.parameters()).data
        
        if(gpu):
            hidden = (weight.new(self.n_layers,batch_size,self.n_hidden).zero_().cuda()\
                      , weight.new(self.n_layers,batch_size,self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers,batch_size,self.n_hidden).zero_()\
                      , weight.new(self.n_layers,batch_size,self.n_hidden).zero_())
            
        return hidden

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharLSTM network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(gpu):
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length))
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


if __name__=="__main__":
    fn = sys.argv[1] 
    with open(fn,'r') as f:
        text = f.read()
    
    
    
    chars = tuple(set(text))
    
    int2char = dict(enumerate(chars))
    
    char2int = {ch:ii for ii,ch in int2char.items()}
    
    encoded = np.array([char2int[ch] for ch in text])
    
    
    n_hidden = 512 
    n_layers = 3
    
    net = CharLSTM(chars,n_hidden,n_layers)
    print(net)
    
    batch_size = 512
    seq_length = 100
    n_epochs = 20
    
    train(net,encoded,epochs=n_epochs,batch_size=batch_size,seq_length=seq_length,lr=0.001,print_every=10)
    
    
    model_name = 'lstm-'+str(n_epochs)+'_epoch-'+fn+'.net'
    
    checkpoint = {
        'n_hidden':net.n_hidden,
        'n_layers':net.n_layers,
        'state_dict':net.state_dict(),
        'tokens':net.chars
    }
    
    with open(model_name,'wb') as f:
        torch.save(checkpoint,f)
    
