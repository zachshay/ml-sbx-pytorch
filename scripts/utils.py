import random

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data(batch_size: int,
              train_split: float = 0.8,
              data_root: str = "./data",
              num_workers: int = 2,
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

    for data, _ in train_loader:
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
    test_set = datasets.MNIS(root = "./data",
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

def train_model(model, device, train_loader, optimizer, epochs):
    """common program to execute training loop"""
    return