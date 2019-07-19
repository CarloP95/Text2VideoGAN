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

from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class Trainer:

    def __init__(self, network, trainLoader, num_epochs = 120, save_interval = 20,
                    device = torch.device('cpu'), testLoader = None, validLoader = None,
                    lossCriterion = CrossEntropyLoss, 
                    optimizer = Adam, lr = 0.001, weight_decay = 1e-4 ):

        self.device         = device
        self.network        = network
        self.trainLoader    = trainLoader
        self.validLoader    = validLoader
        self.testLoader     = testLoader
        self.num_epochs     = num_epochs
        self.save_interval  = save_interval
        self.lossCriterion  = lossCriterion()
        self.optimizer      = optimizer(self.network.parameters(), lr = lr, weight_decay = weight_decay)

        self.moveToDevice()


    def moveToDevice(self):
        
        toMigrate = [self.optimizer, self.lossCriterion, self.network]

        for moveItem in toMigrate:
            moveItem.to(self.device)

    
    def _cycle(self, mode, loader):
        
        isTraining = mode == 'train'

        if isTraining:
            self.network.train()
            torch.set_grad_enabled(True)
        else:
            self.network.eval()
            torch.set_grad_enabled(False)

        losses      = []
        accuracies  = []

        for index, (sentences, actions) in loader:
            
            self.optimizer.zero_grad()

            sentences.to(self.device)
            actions.to(self.device)

            output = self.network(sentences)
            loss = self.lossCriterion(output, actions)

            predictedAction , predicted = output.max(1)
            correct = predicted.eq(sentences).sum().item()
            accuracy = correct/sentences.size(0)

            if isTraining:
                loss.backward()
                self.optimizer.step()

            losses.append(loss)
            accuracies.append(accuracy)

            print(f'\rBatch [{index}/{len(loader)}] {mode.capitalize()}\t Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}', end = '')

        return (losses, accuracies)


    def start(self):
        
        for current_epoch in range(self.num_epochs):
            # Initialize accumulators for computing average loss/accuracy
            epoch_loss_sum = {'train': 0, 'test': 0}
            epoch_loss_cnt = {'train': 0, 'test': 0}
            epoch_accuracy_sum = {'train': 0, 'test': 0}
            epoch_accuracy_cnt = {'train': 0, 'test': 0}
            # Process each split
            for split, loader in [("train", self.trainLoader), ("valid", self.validLoader), ("test", self.testLoader)]:
                
                self._cycle(split, loader)
            # Compute average epoch loss/accuracy
            avg_train_loss = epoch_loss_sum["train"]/epoch_loss_cnt["train"]
            avg_train_accuracy = epoch_accuracy_sum["train"]/epoch_accuracy_cnt["train"]
            avg_test_loss = epoch_loss_sum["test"]/epoch_loss_cnt["test"]
            avg_test_accuracy = epoch_accuracy_sum["test"]/epoch_accuracy_cnt["test"]
            print(f"Epoch: {epoch+1}, TL={avg_train_loss:.4f}, TA={avg_train_accuracy:.4f}, ŦL={avg_test_loss:.4f}, ŦA={avg_test_accuracy:.4f}")

            if current_epoch % self.save_interval == 0:
                self.network.saveState(current_epoch)
        
        return {
             "avg_train_loss" : avg_train_loss,
             "avg_train_accuracy" : avg_train_accuracy, 
             "avg_test_loss" : avg_test_loss, 
             "avg_test_accuracy" : avg_test_accuracy
            }
