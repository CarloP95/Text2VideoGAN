import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

class LSTM(nn.Module):

    def __init__(self, rnn_type, embed_size, rnn_size, optimizer, inputVocabolary):
        # Call parent constructor
        super().__init__()
        'We use the embedding layer because the input dataset will be a vocabulary of many words.'
        self.embedding = nn.Embedding(len(inputVocabolary)+1, embed_size) 
        self.rnn = rnn_type(embed_size, rnn_size, batch_first=True)
        self.output = nn.Linear(rnn_size, 2)
        self.optimizer = optimizer
        
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

    """
     @param trainData is a dictionary with the following structure, that needs a TensorDataset as (train/test)_dataset.
        {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True),
            "test":  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)
        } 
    """
    def trainAndTest(self, num_epochs, device, trainData):
        # Start training
        for epoch in range(num_epochs):
            # Initialize accumulators for computing average loss/accuracy
            epoch_loss_sum = {'train': 0, 'test': 0}
            epoch_loss_cnt = {'train': 0, 'test': 0}
            epoch_accuracy_sum = {'train': 0, 'test': 0}
            epoch_accuracy_cnt = {'train': 0, 'test': 0}
            # Process each split
            for split in ["train", "test"]:
                # Set network mode
                if split == "train":
                    super.train()
                    torch.set_grad_enabled(True)
                else:
                    super.eval()
                    torch.set_grad_enabled(False)
                # Process all data in split
                data_iter = iter(trainData[split])
                data_len = len(trainData[split])
                for i in range(data_len):
                    # Read data
                    input, target = next(data_iter)
                    # Move to device
                    input = input.to(device)
                    target = target.to(device)
                    # Forward
                    output = self(input)
                    loss = F.cross_entropy(output, target)
                    # Update loss sum
                    epoch_loss_sum[split] += loss.item()
                    epoch_loss_cnt[split] += 1
                    # Compute accuracy
                    _,pred = output.max(1)
                    correct = pred.eq(target).sum().item()
                    accuracy = correct/input.size(0)
                    # Update accuracy sum
                    epoch_accuracy_sum[split] += accuracy
                    epoch_accuracy_cnt[split] += 1
                    # Backward and optimize
                    if split == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
            # Compute average epoch loss/accuracy
            avg_train_loss = epoch_loss_sum["train"]/epoch_loss_cnt["train"]
            avg_train_accuracy = epoch_accuracy_sum["train"]/epoch_accuracy_cnt["train"]
            avg_test_loss = epoch_loss_sum["test"]/epoch_loss_cnt["test"]
            avg_test_accuracy = epoch_accuracy_sum["test"]/epoch_accuracy_cnt["test"]
            print(f"Epoch: {epoch+1}, TL={avg_train_loss:.4f}, TA={avg_train_accuracy:.4f}, ŦL={avg_test_loss:.4f}, ŦA={avg_test_accuracy:.4f}")
        
        return {
             "avg_train_loss" : avg_train_loss,
             "avg_train_accuracy" : avg_train_accuracy, 
             "avg_test_loss" : avg_test_loss, 
             "avg_test_accuracy" : avg_test_accuracy
             }


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
