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
                    optimizer = Adam, lr = 0.001, weight_decay = 1e-4,
                    loadEpoch = None ):

        self.device         = device
        self.network        = network
        self.trainLoader    = trainLoader
        self.validLoader    = validLoader
        self.testLoader     = testLoader
        self.num_epochs     = num_epochs
        self.save_interval  = save_interval
        self.lossCriterion  = lossCriterion()
        self.optimizer      = optimizer(self.network.parameters(), lr = lr, weight_decay = weight_decay)
        self.loadEpoch      = loadEpoch

        self.moveToDevice()
        self.load()


    def load(self):
        if self.loadEpoch:
            self.network.loadState(self.loadEpoch)

    def moveToDevice(self):
        
        toMigrate = [self.lossCriterion, self.network]

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

        for index, (sentences, actions) in enumerate(loader):
            
            self.optimizer.zero_grad()

            sentences = sentences.to(self.device)
            actions   = actions.to(self.device);    actions -= 1

            output = self.network(sentences)
            loss = self.lossCriterion(output, actions)

            predictedValues , predictedActions = output.max(1)
            correct = predictedActions.eq(actions).sum().item()
            accuracy = correct/sentences.size(0)

            if isTraining:
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())
            accuracies.append(accuracy)

            print(f'\rBatch [{index + 1}/{len(loader)}] {mode.capitalize()}\t Loss: {loss:.4f}, Accuracy: {accuracy:.4f}', end = '')

        return (losses, accuracies)


    def start(self):


        firstEpoch_overfit      = False
        earlyStop               = False


        for current_epoch in range(self.num_epochs):
            # Initialize accumulators for computing average loss/accuracy
            epoch_loss_sum = {'train': 0, 'valid':0, 'test': 0}
            epoch_loss_cnt = {'train': 0, 'valid':0, 'test': 0}
            epoch_accuracy_sum = {'train': 0, 'valid':0, 'test': 0}
            epoch_accuracy_cnt = {'train': 0, 'valid':0, 'test': 0}
            

            if earlyStop and not firstEpoch_overfit:
                prevEpochNetworkState   = self.network.state_dict()
                prevEpochOptimizerState = self.optimizer.state_dict()
            
            print(f'{"-"*10}Epoch {current_epoch + 1}{"-"*10}')
            # Process each split
            for split, loader in [("train", self.trainLoader), ("valid", self.validLoader), ("test", self.testLoader)]:
                
                losses, accuracies = self._cycle(split, loader)
                print('\n')

                epoch_loss_cnt[split]       += 1
                epoch_loss_sum[split]       += sum(losses)/len(losses)
                epoch_accuracy_cnt[split]   += 1
                epoch_accuracy_sum[split]   += sum(accuracies)/len(accuracies)


            # Compute average epoch loss/accuracy
            avg_train_loss      = epoch_loss_sum["train"]/epoch_loss_cnt["train"]
            avg_train_accuracy  = epoch_accuracy_sum["train"]/epoch_accuracy_cnt["train"]
            avg_valid_loss      = epoch_loss_sum["valid"]/epoch_loss_cnt["valid"]
            avg_valid_accuracy  = epoch_accuracy_sum["valid"]/epoch_accuracy_cnt["valid"]
            avg_test_loss       = epoch_loss_sum["test"]/epoch_loss_cnt["test"]
            avg_test_accuracy   = epoch_accuracy_sum["test"]/epoch_accuracy_cnt["test"]

            print(f"Epoch: {current_epoch + 1}/{self.num_epochs}, Train: Loss={avg_train_loss:.4f}, Accuracy ={avg_train_accuracy:.4f}.\
                Validation: Loss={avg_valid_loss:.4f}, Accuracy={avg_valid_accuracy:.4f}.\
                Test: Loss={avg_test_loss:.4f}, Accuracy={avg_test_accuracy:.4f}.\n")
            
            ## Check for overfitting
            if earlyStop:
                
                if avg_train_accuracy > avg_valid_accuracy and avg_train_loss < avg_valid_loss:

                    if not firstEpoch_overfit:
                        firstEpoch_overfit = True
                        print('Probable overfit is occurring. Next epoch weights will be loaded and learning rate will be updated.')

                    else:
                        firstEpoch_overfit = False
                        self.network.load_state_dict(prevEpochNetworkState)
                        self.optimizer.load_state_dict(prevEpochOptimizerState)
                        for g in self.optimizer.param_groups:
                            g['lr'] = g['lr']/10
                        print('Overfit Detected. Loading back 2 epochs ago and reducing learning rate.')
                
                else: # Reset, second epoch overfit was not detected.
                    firstEpoch_overfit = False if firstEpoch_overfit else True

            if int(current_epoch + 1) % int(self.save_interval) == 0:
                print('Saving the state...')
                self.network.saveState(current_epoch + 1)
        

        return {
             "avg_train_loss" : avg_train_loss,
             "avg_train_accuracy" : avg_train_accuracy, 
             "avg_test_loss" : avg_test_loss, 
             "avg_test_accuracy" : avg_test_accuracy
            }
