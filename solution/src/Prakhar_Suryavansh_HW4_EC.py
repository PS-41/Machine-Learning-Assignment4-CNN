import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pickle

'''
Homework: Image Classification using CNN
Instructions:
1. Follow the skeleton code precisely as provided.
2. You may define additional helper functions if necessary, but ensure the input/output format is maintained.
3. Visualize and report results as specified in the problem.
'''

# (a) Dataloader: Download the MNIST dataset and get the dataloader in PyTorch
train_data = None  # Placeholder for train data
val_data = None  # Placeholder for validation data

# (c) Modeling: Implement a Convolutional Neural Network
class CNNModel(nn.Module):
    '''
    CNN Model for image classification.    
    '''
    def __init__(self):
        super().__init__()
        # Define the CNN layers

    def forward(self, x):
        # Forward pass through the network
        return x

# (1) Training and Evaluation
# Students should define the optimizer and scheduler outside this function if needed
def train_and_evaluate(model, train_loader, val_loader, epochs=5):
    '''
    Train the model and evaluate on the validation set.

    Parameters:
        model (CNNModel): The CNN model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        epochs (int): Number of training epochs.
    Returns:
        float: Validation accuracy.
    '''

# (4) Hyper-parameter tuning function
# This function should not use excessive resources and should be efficient for a sandbox environment
def tune_hyperparameters(train_loader, val_loader):
    '''
    Tune hyperparameters of the CNN model using the validation set.

    Parameters:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
    Returns:
        CNNModel: The best model.
    '''
    best_model = None  # Placeholder for the best model
    return best_model

if __name__ == "__main__":
    # Obtain train_loader and val_loader here
    # train_loader, val_loader = DataLoader definitions outside this code block

    # Tune hyperparameters and find the best model
    best_model = tune_hyperparameters(train_loader, val_loader)

    # Save the best model to a pickle file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
