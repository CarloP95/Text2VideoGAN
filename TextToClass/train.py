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

import torch
import argparse
import torch.nn as nn

from models import LSTM
from trainer import Trainer
from dataloading import TextLoader, DataLoaderFactory
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn; cudnn.benchmark = True



def addCLArguments(parser):

    parser.add_argument('--cuda', default = False, action= 'store_true', 
                            help= 'Set to use the GPU.')
                            
    parser.add_argument('--epochs', default = 120, type = int,
                            help= 'Set the number of epochs to train.')

    parser.add_argument('--numClasses', default = 101, type = int,
                            help= 'Set the number of classes that the model must predict.')

    parser.add_argument('--batch_size', default = 64, type = int,
                            help= 'Set the number that will be used as batch size.')

    parser.add_argument('--path', default = 'caffe/examples/s2vt/results/dataset_Action_Description.txt', type = str,
                            help= 'Set the relative path to find the file that contains the dataset.')

    parser.add_argument('--embed_size', default = 256, type = int,
                            help= 'Set the embedding size for the RNN.')

    parser.add_argument('--rnn_size', default = 512, type = int,
                            help= 'Set the RNN size.')

    parser.add_argument('--lr', default = 1e-4, type = float,
                            help= 'Set the learning rate for the optimizer.')

    parser.add_argument('--weight_decay', default = 5e-4, type = float,
                            help= 'Set the weight decay for the optimizer.')

    parser.add_argument('--load_epoch', default = 0, type = int,
                            help= 'Set the state epoch to load from disk.')

    parser.add_argument('--save_interval', default = 20, type = int,
                            help= 'Set save interval for the model state.')

    parser.add_argument('--sequence_length', default = 30, type = int,
                            help= 'Set the maximum length for each item that will be given to the model.')


    return parser


def getCLArguments(parser):
    args = parser.parse_args()

    return {
        'cuda'          : args.cuda,
        'epochs'        : args.epochs,
        'numClasses'    : args.numClasses,
        'path'          : args.path,
        'batch_size'    : args.batch_size,
        'weight_decay'  : args.weight_decay,
        'lr'            : args.lr,
        'rnn_size'      : args.rnn_size,
        'embed_size'    : args.embed_size,
        'loadEpoch'     : args.load_epoch,
        'save_interval' : args.save_interval,
        'sequence_len'  : args.sequence_length
    }


def getDevice(clParameters):

    availableAndDesired = clParameters['cuda'] and torch.cuda.is_available()
    print(f'The training will happen on {"CPU" if not availableAndDesired else "GPU"}.')

    return torch.device('cuda') if availableAndDesired else torch.device('cpu')


if __name__ == "__main__":

    clParameters    = getCLArguments( addCLArguments(argparse.ArgumentParser()) )
    device          = getDevice(clParameters)

    dataset         = TextLoader(clParameters['path'], item_length= clParameters['sequence_len'])
    factory         = DataLoaderFactory(dataset, clParameters['batch_size'])
    train_dataLoader, validation_dataLoader, test_dataLoader = factory.dataloaders

    network = LSTM(nn.LSTM, clParameters['rnn_size'], clParameters['embed_size'], dataset.vocabulary)
    trainer = Trainer(network, train_dataLoader, clParameters['epochs'], device = device,
                        testLoader= test_dataLoader, validLoader= validation_dataLoader,
                        lr= clParameters['lr'], weight_decay= clParameters['weight_decay'], 
                        loadEpoch= clParameters['loadEpoch'], save_interval= clParameters['save_interval'] )
    try:
        trainer.start()

    except KeyboardInterrupt:
        pass

    network             = trainer.network

    while(True):

        text                = input('Input a string to test how does the model performs >')
        tensor              = torch.tensor(dataset.prepareTxtForTensor(text)).cuda().unsqueeze_(0)
        output              = network(tensor)
        probability, action = output.max(1)
        
        print(f'Predicted class is {dataset.getClassNameFromIndex(action)} with probability {probability}')


    

