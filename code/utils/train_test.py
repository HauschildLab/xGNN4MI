import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import gc
import os
import GNN_patch
from sklearn.metrics import accuracy_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model: GNN_patch, train_loader:DataLoader, optimizer, criterion):
    """
    Train the model
    
    :param model: GNN model
    :param train_loader: Loader of the dataset
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param scheduler: Learning rate scheduler
    """
    model.train()
    
    total_loss = 0
    correct = 0
    
    for i, data in enumerate(train_loader, 0):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        pred = out.argmax(dim=1)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        correct += int((pred == data.y).sum())        
        
        del data
        gc.collect()
        torch.cuda.empty_cache()

    accuracy = correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)  
    return avg_loss, accuracy
        


def test_model(model, loader, criterion = None):
    """
    Test the model
    
    :param loader: Loader of the dataset
    :param model: GNN model
    :param criterion: Loss function
    
    :return: test loss, test accuracy, list of predicted labels, list of true labels
    """
    
    model.eval()
    
    correct = 0
    running_loss = 0.0
    steps = 0
    
    y_test=[]
    y_pred=[] #list to keep predicted labels
    y_prob=[] #list to keep predicted probabilities
    
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            data = data.to(device)
            out = model(data)
            if criterion:
                loss = criterion(out, data.y)
                running_loss += loss.item()
            else:
                loss = 0
                running_loss = 0.0
            pred = out.argmax(dim=1) #Use class with highest probability
            probs = torch.nn.functional.softmax(out, dim=1) # Convert logits to probabilities
            steps += 1
            correct += int((pred == data.y).sum())
            y_pred.extend(list(pred.detach().cpu().numpy()))
            y_test.extend(list(data.y.detach().cpu().numpy()))
            y_prob.extend(list(probs.detach().cpu().numpy()))
            
            del data
            gc.collect()
            torch.cuda.empty_cache()
            
    accuracy = correct / len(loader.dataset)
    avg_loss = running_loss / steps  
    return avg_loss, accuracy, y_pred, y_test, y_prob
        