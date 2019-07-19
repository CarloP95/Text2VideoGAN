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
import torch.optim as optim
import torch.nn.functional as F

from time import time
from dataloading import TextLoader, DataLoaderFactory
from torch.utils.data import DataLoader



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

    return parser


def getCLArguments(parser):
    args = parser.parse_args()

    return {
        'cuda'          : args.cuda,
        'epochs'        : args.epochs,
        'numClasses'    : args.numClasses,
        'path'          : args.path,
        'batch_size'    : args.batch_size
    }


def getDevice(clParameters):

    availableAndDesired = clParameters['cuda'] and torch.cuda.is_available()
    print(f'The training will happen on {"CPU" if not availableAndDesired else "GPU"}.')

    return torch.device('cuda') if availableAndDesired else torch.device('cpu')


if __name__ == "__main__":

    clParameters    = getCLArguments( addCLArguments(argparse.ArgumentParser()) )
    device          = getDevice(clParameters)

    factory         = DataLoaderFactory(TextLoader(clParameters['path']), clParameters['batch_size']) 
    train_dataLoader, validation_dataLoader, test_dataLoader = factory.dataloaders


    

