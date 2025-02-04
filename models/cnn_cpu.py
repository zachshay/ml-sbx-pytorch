import torch
import torch.nn as nn
from scripts.utils import load_data, train_model

class CNN(nn.module):
    # ... (CNN layers) ...
    pass

device = torch.device("cpu")

train_loader, test_loader = load_data(batch_size = 32)

model = CNN().to(device)

# train_model(model, device, train_loader, optimizer, epochs = 10)