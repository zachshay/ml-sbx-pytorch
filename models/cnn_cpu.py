import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils import load_data, train_model

class CNN(nn.Module):
    # initialize the class
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        # pooling layers
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # fully connected layers
        self.fc1 = nn.Linear(in_features = self._calculate_flatten_size(), out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 10)
    
    def _calculate_flatten_size(self):
        # create dummy input
        dummy_input = torch.randn(1, 1, 28, 28)

        x = self.forward_features(dummy_input)

        return x.view(-1).size(0)
    
    def forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.fc2(x)

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
print(f"Training run for CNN on CPU.")
print(f"  * Best Validation Loss: {best_val_loss: .4f}")
print(f"  * Best Validation Accuracy: {best_val_acc: .2f}")
print(f"  * Training Time: {eleapsed_time: .2f} seconds")