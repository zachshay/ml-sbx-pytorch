import sys
import time

import torch
import torch.nn as nn

from scripts.utils import load_data, train_model

class CNN(nn.Module):
    # initialize the class
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc = nn.Linear(1000 , 10)  # 1000 is only a placeholder value
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        x = torch.flatten(x, start_dim = 1)
        
        # bug: dynamically adjust the input size of the fully connected layer
        if(self.fc.in_features != x.shape):
            # print(f"This is x.shape[1]: {x.shape[1]}")
            self.fc = nn.Linear(x.shape[1], 10)
            # print(f"Adjusted FC layer input size to: {x.shape}")

        x = self.fc(x)

        return x

# choose a device
device = torch.device("cpu")

# load the data
train_loader, val_loader, test_loader, mean, std = load_data(batch_size = 64)

# create the model
model = CNN().to(device)

# define remaining helpers
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# record the start time
start_time = time.time()

# train/optimize the model
model, best_val_loss, best_val_acc = train_model(model, device, train_loader, val_loader, optimizer, criterion, epochs = 10)

# record the end time
end_time = time.time()

# calculate the duration
eleapsed_time = end_time - start_time

# save the model to disk
torch.save(model.state_dict(), "cnn_cpu.pth")

# print the final metrics
print(f"Best Validation Loss: {best_val_loss: .4f}")
print(f"Best Validation Accuracy: {best_val_acc: .2f}")
print(f"Training Time: {eleapsed_time: .2f} seconds")