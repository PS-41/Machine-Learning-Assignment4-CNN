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
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std for MNIST
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# Define DataLoaders for training and validation sets
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# (c) Modeling: Implement a Convolutional Neural Network
class CNNModel(nn.Module):
    '''
    CNN Model for image classification.    
    '''
    def __init__(self):
        super().__init__()
        # Define the CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After pooling, each feature map is 7x7
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for digits 0-9

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# (1) Training and Evaluation
# Students should define the optimizer and scheduler outside this function if needed
def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler=None, epochs=5):
    '''
    Train the model and evaluate on the validation set.

    Parameters:
        model (CNNModel): The CNN model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        epochs (int): Number of training epochs.
    Returns:
        float: Validation accuracy.
    '''
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Update learning rate if a scheduler is provided
        if scheduler:
            scheduler.step()
        
        # Validation step
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return val_accuracy

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
    best_accuracy = 0
    best_model = None  # Placeholder for the best model
    
    # Loop through hyperparameter combinations
    for lr in [0.001, 0.0001]:  # Learning rates to try
        for batch_size in [32, 64]:  # Batch sizes to try
            # Define model, optimizer, and optionally a scheduler
            model = CNNModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            # Update dataloader with new batch size
            train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False)

            # Train and evaluate model
            accuracy = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler)

            # Update best model if accuracy is improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
    return best_model
    
if __name__ == "__main__":
    # Obtain train_loader and val_loader here
    # train_loader, val_loader = DataLoader definitions outside this code block

    # Tune hyperparameters and find the best model
    best_model = tune_hyperparameters(train_loader, val_loader)

    # Save the best model to a pickle file
    with open('Prakhar_Suryavansh_Model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
