##########################################################################################
## MIT License                                                                          ##
##                                                                                      ##
## Copyright (c) [2019] [ CarloP95 carlop95@hotmail.it,                                 ##
##                        vitomessi vitomessi93@gmail.com ]                             ##
##                                                                                      ##
##                                                                                      ##
## Permission is hereby granted, free of charge, to any person obtaining a copy         ##
## of this software and associated documentation files (the "Software"), to deal        ##
## in the Software without restriction, including without limitation the rights         ##
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            ##
## copies of the Software, and to permit persons to whom the Software is                ##
## furnished to do so, subject to the following conditions:                             ##
##                                                                                      ##
## The above copyright notice and this permission notice shall be included in all       ##
## copies or substantial portions of the Software.                                      ##
##                                                                                      ##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           ##
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             ##
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          ##
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               ##
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        ##
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        ##
## SOFTWARE.                                                                            ##
##########################################################################################

import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, rnn_type, rnn_size, embed_size, inputVocabolary, numClasses = 101,
                    checkpoint = "LSTM-checkpoint.pth"):

        super(LSTM, self).__init__()
        'We use the embedding layer because the input dataset will be a vocabulary of many words.'
        self.embedding = nn.Embedding(len(inputVocabolary)+1, embed_size) 
        self.rnn = rnn_type(embed_size, rnn_size, batch_first=True)
        self.output = nn.Linear(rnn_size, numClasses)
        self.checkpoint = checkpoint

        
    def forward(self, x):
        # Embed data
        x = self.embedding(x)
        # Process through RNN
        x,_ = self.rnn(x)
        # Get final state
        # - x is (batch, time, features)
        # - we want, for all batches, the features at the last time step
        # - x[:,-1,:] -> (batch, features)
        x = x[:,-1,:]
        # Classify
        x = self.output(x)
        return x


    def loadState(self, epoch):
        state_dict = torch.load(self.checkpoint[:-4] +  f'-{epoch}.pth')
        self.load_state_dict(state_dict)


    def saveState(self, epoch):
        state_dict = self.state_dict()
        for k,v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, self.checkpoint[:-4] + f"-{epoch}.pth")

