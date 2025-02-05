import random
import sys

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

def load_data(batch_size: int,
              train_split: float = 0.8,
              data_root: str = "./data",
              num_workers: int = 0,         # anything other than 0 crashes
              seed: int = 42) -> tuple:
    """Loads and preprocesses MNIST data

    Calculates the mean and standard deviation of the training set, and then
    applies transformations to convert data into tensors & normalize it using
    the calculated statistics.

    Args:
        batch_size (int): the desired batch size for DataLoaders
        train_spit (float, optional):   fraction of training data set to be used
                                        for training; the remainder is used for validating
        data_root (str, optional):  directory used to store downloaded data
        num_workers (int, optional):    number of worker processes to use for data loading
        seed (int, optional):   random seed for reproducibility
    
    Returns: tuple
        - train_loader (DataLoader): for the training set
        - val_loader (DataLoader): for the validation set
        - test_loader (DataLoader): for the test set
        - mean (float): the mean of the training set
        - std_dev (float): the std_dev of the training set
    """

    # Bug: set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if(torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        torch.mps.manual_seed(seed)
    
    # Consideration: disable benchmark if using CUDA (not applicable here)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Boostrap the data: download & convert to tensors
    train_val_set = datasets.MNIST(root = "./data",
                                   train = True,
                                   download = True,
                                   transform = transforms.ToTensor())
    
    # Calculate the size of my trianing & validation sets
    train_size = int(train_split * len(train_val_set))          # used to train models
    val_size = len(train_val_set) - train_size                  # used to compare models (tuning)

    # Split the data (a mask; not a copy of subsets)
    train_set, val_set = random_split(train_val_set, [train_size,
                                                      val_size])
    
    # Use the batch_size to avoid OOM errors
    my_batch_size = batch_size

    # Configure a loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = my_batch_size,
                                               shuffle = False,             # important for mean/std. dev. calcs
                                               num_workers = num_workers
                                               )
    
    # Calculate the mean, standard deviation (for normalization)
    mean = 0.0
    std_dev = 0.0
    num_samples = 0

    if(train_loader is None):
        print("Why is this None?")
        sys.exit(1)
    else:
        # print("Train Loader is not None.")
        # print(f"It has a length of {len(train_loader)}")
        # for i, (data, labels) in enumerate(train_loader):
        #     if(i > 2):
        #         sys.exit(1)
        #     else:
        #         print(f"Batch {i+1}")
        #         print(f"Data is shaped: {data.shape}")
        #         print(f"Labels is shape: {labels.shape}")
        #         print(data)
        #         print(labels)
        pass

    for data, _ in train_loader:
        # print(data)
        # print(_)
        # sys.exit(1)

        # document this batch's details
        batch_mean = data.mean()
        batch_std = data.std()
        batch_size = data.size(0)

        # increment the larger set's details
        mean += (batch_mean * batch_size)
        std_dev += (batch_std * batch_size)
        num_samples += batch_size
    
    # finalize the calculation
    mean = (mean / num_samples)
    std_dev = (std_dev / num_samples)

    # Add normalization to list of transformations
    normalization_seq = transforms.Compose([
                                            transforms.ToTensor(),            # bug: earlier transformation is lost on reload
                                            transforms.Normalize((mean, ),
                                                                 (std_dev, )
                                                                 )
                                            ])
    
    # Set the appropriate transformation to the training & validation sets
    train_set.dataset.transform = normalization_seq
    val_set.dataset.transform = normalization_seq

    # Create the test set
    test_set = datasets.MNIST(root = "./data",
                             train = False,
                             download = True,
                             transform = normalization_seq)
    
    # Configure the final loaders to be returned for all sets
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = my_batch_size,
                                               shuffle = True,
                                               num_workers = num_workers)
    
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size = my_batch_size,
                                             shuffle = False,
                                             num_workers = num_workers)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size = my_batch_size,
                                              shuffle = False,
                                              num_workers = num_workers)
    
    # Return the expected tuple
    return (train_loader,
            val_loader,
            test_loader,
            mean,
            std_dev)

def train_model(model: nn.Module,
                device: torch.device,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                epochs: int,
                log_dir: str) -> dict:
    """Trains a PyTorch model and evaluates it on a validation set.

    Args:
        model (nn.Module): the PyTorch model to train
        device (torch.device): the device used to train the model (e.g., 'cpu' or 'mps')
        train_loader (DataLoader): DataLoader for the training set
        val_loader (DataLoader): DataLoader for the validation set
        optimizer (torch.optim.Optimizer): the optimizer to use for training
        criterion (nn.Module): the loss function to use for training
        epochs (int): the number of epochs for training
        log_dir (str): the location to write experiement logs
    
    Returns: tuple
        model (nn.Module): the optimized model
        val_loss (float): the final validation loss
        val_acc (float): the final validation accuracy
    """

    # Create important placeholders
    best_model_state_dict = None    # state_dict is far more efficient than a whole copy
    best_loss = -1.0                # should never be negative (bootstraps nicely)
    best_acc = -1.0                 # should never be negative (boostraps nicely)

    # Create the tensorboard writer
    writer = SummaryWriter(log_dir)

    # Need a dictionary to track historical values
    history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []}
    
    # begin...  The Training Loop

    for epoch in range(epochs):
        ###
        # Training Phase
        ###

        model.train()

        # Initialize Important Placeholders
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # process batches of training data
        for i, (inputs, labels) in enumerate(train_loader):
            # move data to the right device
            inputs, labels = inputs.to(device), labels.to(device)

            # zero-out the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)             # forward pass (input -> model -> prediction)
            loss = criterion(outputs, labels)   # calculate loss
            loss.backward()                     # backward pass (calculate gradients based on loss)
            optimizer.step()                    # update model parameters using calculated gradient

            train_loss += loss.item()                                   # capture this loop's loss
            max_value, predicted_class = torch.max(outputs.data, 1)     # get predicted label (highest value within row)
            train_total += labels.size(0)                               # accumulate number of predictions
            train_correct += (predicted_class == labels).sum().item()   # accumulate correct predictions

            writer.add_scalar("Loss/train/batch", loss.item(), epoch * len(train_loader) + i)   # write batch data to tensorboard
        
        # calculate key details of this epoch's training model (on training data)
        train_loss /= len(train_loader)                 # calculate this epoch's average training loss
        train_acc = 100 * (train_correct / train_total) # calculate this epoch's training accuracy

        # write epoch data to tensorboard
        writer.add_scalar("Loss/train/epoch", train_loss, epoch)



        ###
        # Validation Phase
        ###

        model.eval()

        # initialize important placeholders
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # disable gradient calculation (no backwards pass for inference-only use)
        with torch.no_grad():
            # process batches of validation data
            for inputs, labels in val_loader:
                # move data to the right device
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)             # forward pass (input -> model -> prediction)
                loss = criterion(outputs, labels)   # calculate loss

                val_loss += loss.item()                                 # capture this loop's loss
                max_value, predicted_class = torch.max(outputs.data, 1) # get predicted label (highest value within row)
                val_total += labels.size(0)                             # accumulate number of predictions
                val_correct += (predicted_class == labels).sum().item() # accumulate correct predictions
            
            # calculate key details of this epoch's training model (on validation data)
            val_loss /= len(val_loader)                 # calculate this epoch's average validation loss
            val_acc = 100 * (val_correct / val_total)   # calculate this epoch's validation accuracy

            # Write validation data to tensorboard
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
        
        # Report this epoch's critical metrics
        print(f"Epoch [{epoch + 1} / {epochs}], "
              f"Train Loss: {train_loss: .4f}, Train Acc. {train_acc: .2f}%, "
              f"Val Loss: {val_loss: .4f}, Val Acc.: {val_acc: .2f}%")
        
        # Document this epoch's critical metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update placeholders if merited
        if(val_acc > best_acc):
            best_acc = val_acc
            best_loss = val_loss
            best_model_state_dict = model.state_dict()
    
    ###
    # Training Loop Complete
    ###

    # Close the SummaryWriter
    writer.close()

    # just to be safe...  check for none 
    if(best_model_state_dict is None):
        print("Error: train_model() -- best_model_state_dict should not be None")
        return (None, best_loss, best_acc)
    else:
        # Adjust the model
        model.load_state_dict(best_model_state_dict)

        # create the expected tuple
        tuple_toReturn = (
                            model,
                            best_loss,
                            best_acc)
        
        # return the tuple
        return tuple_toReturn 