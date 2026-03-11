"""
Train a Graph Neural Network (GNN) for PTB-XL diagnostic superclass classification.

By default, the model is retrained on the combined training and validation sets
and evaluated on the test set. Optionally, 5-fold cross-validation can be enabled
using the --cv flag.

Environment variables:
    DATASET_LOCATION : path to the PTB-XL dataset directory
    SAVE_LOCATION    : directory where results and model checkpoints will be stored
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.loader import DataLoader
import gc
from ecg_to_graph_patch import GraphDataset
import argparse
import os
from torch_geometric import seed_everything
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import ast

from utils.train_test import train_model, test_model

#Implementing Argparse to get commands in terminal:
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('-p', '--num_patches', type=int, default=10, help='Number of patches')
parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation instead of retraining")
args = parser.parse_args()

epochs = int(args.epochs)
batch_size = int(args.batch_size)
lr = float(args.learning_rate)
num_patches = int(args.num_patches)


if num_patches == 10:
    from GNN_patch10 import GCN_10
elif num_patches == 25:
    from GNN_patch25 import GCN_25
elif num_patches == 50:
    from GNN_patch50 import GCN_50
elif num_patches == 100:
    from GNN_patch100 import GCN_100


# Get data
data_dir = os.environ.get('DATASET_LOCATION')
save_dir = os.environ.get('SAVE_LOCATION','.')
path = os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
dataset = GraphDataset(root=path, num_patches=num_patches)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Split the dataset into training and testing sets
seed_everything(42)
num_samples = len(dataset)

#Filtering for train, val and test:
df = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
dg = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col = 0)
agg_df = dg[dg.diagnostic == 1]
        
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    if tmp:
        return tmp[0]
    else:
        return 'nicht_bestatigt'

df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)

df_new = df[df.diagnostic_superclass != 'nicht_bestatigt']
df_new = df_new[df_new.validated_by_human == 1]

train_mask = df_new.strat_fold <= 8
val_mask = df_new.strat_fold == 9
test_mask = df_new.strat_fold == 10

train_dataset =dataset[np.array(train_mask)]
val_dataset = dataset[np.array(val_mask)]
test_dataset = dataset[np.array(test_mask)]

#for retraining:
retrain_dataset = dataset[np.array(train_mask) | np.array(val_mask)]


# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
num_nodes = 12 * num_patches  # Number of nodes based on the patch size
if num_patches == 10:
    model = GCN_10(dataset, num_nodes, num_patches)
elif num_patches == 25:
    model = GCN_25(dataset, num_nodes, num_patches)
elif num_patches == 50:
    model = GCN_50(dataset, num_nodes, num_patches)
elif num_patches == 100:
    model = GCN_100(dataset, num_nodes, num_patches)
model = model.to(device)

#############################################################################################################
#############################################################################################################
#Training the model:
#############################################################################################################
#############################################################################################################

if not args.cv:
    #Choose an optimizer and loss function:
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        print(f'Epoch: {epoch+1:03d}') 
        #train_model(model, train_loader, optimizer, criterion)
        #train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion)
        #for retraining use:
        train_loss, train_accuracy = train_model(model, retrain_loader, optimizer, criterion)  
            
        print(f'Epoch: {epoch+1:03d}', f'Loss: {train_loss:.4f}')
        
        checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': None,
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': None
                }
        torch.save(checkpoint, os.path.join(save_dir,f'best_model_p{num_patches}_b{batch_size}_lr{lr}_e{epochs}.pt'))

            

    #Testing the model on validation set:
    test_loss, test_accuracy, y_pred, y_test, y_prob = test_model(model, test_loader)



    #############################################################################################################
    #############################################################################################################
    #Training the model with KFold cross-validation:
    #############################################################################################################
    #############################################################################################################
    #Set number of folds
else:
    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold: {fold+1}/{k}')
        
        #Create train and validation datasets for current fold:
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)
        
        #Create train and validation loaders for current fold:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        
        #Initialize the model:
        if num_patches == 10:
            model = GCN_10(dataset, num_nodes, num_patches)
        elif num_patches == 25:
            model = GCN_25(dataset, num_nodes, num_patches)
        elif num_patches == 50:
            model = GCN_50(dataset, num_nodes, num_patches)
        elif num_patches == 100:
            model = GCN_100(dataset, num_nodes, num_patches)

        model = model.to(device)
        
        #Choose an optimizer and loss function:
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        #set the learning rate scheduler
        #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        #Training and testing the model:
        best_val_loss = float('inf')
        patience = 10
        stopping_time = 0

        for epoch in range(epochs):
            print(f'Epoch: {epoch+1:03d}') 
            #train_model(model, train_loader, optimizer, criterion)
            train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion)  
            
            print(f'Epoch: {epoch+1:03d}', f'Loss: {train_loss:.4f}')
            
            #Validate the model:
            val_loss, val_accuracy, y_pred_val, y_test_val = test_model(model, val_loader, criterion)
            
            #Call the learning rate scheduler:
            scheduler.step(val_loss)
            
            #check if the validation loss is decreasing:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stopping_time = 0
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, os.path.join(save_dir,f'best_model_fold{fold+1}_p{num_patches}_b{batch_size}_lr{lr}.pt'))
            
            else:
                stopping_time += 1
                if stopping_time > patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            print(f'Epoch: {epoch+1:03d}', f'Val Loss: {val_loss:.4f}')
            
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_train_accuracies.append(train_accuracy)
        all_val_accuracies.append(val_accuracy)
        
    #Calculate the average loss and accuracy for all folds:
    avg_train_loss = np.mean(all_train_losses)
    avg_val_loss = np.mean(all_val_losses)
    avg_train_accuracy = np.mean(all_train_accuracies)
    avg_val_accuracy = np.mean(all_val_accuracies)

    print(f'Average train loss: {avg_train_loss:.4f}')
    print(f'Average val loss: {avg_val_loss:.4f}')
    print(f'Average train accuracy: {avg_train_accuracy:.4f}')
    print(f'Average val accuracy: {avg_val_accuracy:.4f}')
        
        
