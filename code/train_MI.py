"""

Train a Graph Neural Network (GNN) for myocardial infarction (MI) classification on the PTB-XL dataset.

By default, the model is retrained on the combined training and validation sets and evaluated on the test set.
Optionally, 5-fold cross-validation can be performed using the --cv flag.

Different signal sampling rates can be selected using --sr (paper, 100, or 500 Hz).

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
#from ecg_to_graph_patch_MI import GraphDataset
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
parser.add_argument("--sr", choices=["paper", "100", "500"], default="paper", help="Sampling rate variant for MI graphs.")

args = parser.parse_args()
if args.sr == "paper":
    from ecg_to_graph_patch_MI import GraphDataset
    from GNN_patch25_MI import GCN_25
    run_tag = ""
elif args.sr == "100":
    from ecg_to_graph_patch_MI_100 import GraphDataset
    from GNN_patch25_MI_100Hz import GCN_25
    run_tag = "_100Hz"
else:
    from ecg_to_graph_patch_MI_500 import GraphDataset
    from GNN_patch25_MI_500Hz import GCN_25
    run_tag = "_500Hz"

epochs = int(args.epochs)
batch_size = int(args.batch_size)
lr = float(args.learning_rate)
num_patches = int(args.num_patches)




# Get data
data_dir = os.environ.get('DATASET_LOCATION')
save_root = os.environ.get('SAVE_LOCATION', '.')
save_dir = os.path.join(save_root, 'MI_res')
#print(data_dir)
path = os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
#mi_labels = ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "PMI", "NORM"]
mi_labels = ["IMI", "ASMI", "NORM"]
dataset = GraphDataset(root=path, num_patches=num_patches, mi_labels=mi_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Split the dataset into training and testing sets
seed_everything(42)


# Load metadata
df = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))


#Extract one MI disease label per sample (same logic as in dataset processing)
def extract_disease_label(scp_dict, allowed_labels):
    matches = [code for code in scp_dict.keys() if code in allowed_labels]
    return matches[0] if matches else None

######## Without Norm:
# df['disease_label'] = df['scp_codes'].apply(lambda x: extract_disease_label(x, mi_labels))
# df_new = df[df['disease_label'].notnull()]
# df_new = df_new[df_new['validated_by_human'] == 1]




######## With Norm:
# Extract disease labels from scp_codes
df['disease_label'] = df['scp_codes'].apply(lambda x: extract_disease_label(x, mi_labels))
df = df[df['disease_label'].notnull()]  # Remove rows where no MI/NORM label was found

# Filter to human-validated only
df = df[df['validated_by_human'] == 1]

# Create age groups for balancing
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 120], labels=['<30', '30-45', '45-60', '60+'])

# Split MI and NORM sets
df_norm = df[df['disease_label'] == 'NORM']
df_mi   = df[df['disease_label'] != 'NORM']
df_imi  = df[df['disease_label'] == 'IMI']

# Match NORM patients per fold based only on IMI
def match_norm_to_imi_in_fold(fold_number):
    imi_fold = df_imi[df_imi['strat_fold'] == fold_number]
    norm_fold = df_norm[df_norm['strat_fold'] == fold_number]
    matched_ids = []

    for (sex, age_group), imi_group in imi_fold.groupby(['sex', 'age_group']):
        norm_candidates = norm_fold[(norm_fold['sex'] == sex) & (norm_fold['age_group'] == age_group)]
        sample_size = min(len(imi_group), len(norm_candidates))
        matched = norm_candidates.sample(n=sample_size, random_state=fold_number)
        matched_ids.extend(matched.index.tolist())

    print(f"[Fold {fold_number}] Matched {len(matched_ids)} NORM to {len(imi_fold)} IMI patients")
    return matched_ids

# Perform matching across all folds
matched_norm_ids = []
for fold in range(1, 11):
    matched_norm_ids.extend(match_norm_to_imi_in_fold(fold))

# Final NORM subset (subsampled)
df_norm_matched = df_norm.loc[matched_norm_ids]

# Final dataset: all MI + matched NORM
df_new = pd.concat([df_mi, df_norm_matched])
df_new['disease_label'] = df_new['disease_label'].astype(str)


# Get lists of ECG IDs for each fold
train_ids = df_new[df_new.strat_fold <= 8].index.values
val_ids = df_new[df_new.strat_fold == 9].index.values
test_ids = df_new[df_new.strat_fold == 10].index.values

#Filter the dataset based on ecg_id field in graph data
train_dataset = [data for data in dataset if data.ecg_id.item() in train_ids]
val_dataset = [data for data in dataset if data.ecg_id.item() in val_ids]
test_dataset = [data for data in dataset if data.ecg_id.item() in test_ids]
#for retraining the model:
retrain_dataset = [data for data in dataset if data.ecg_id.item() in train_ids or data.ecg_id.item() in val_ids]

# Optional: print stats
print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

retrain_loader = DataLoader(retrain_dataset, batch_size=batch_size, shuffle=True)

num_classes = len(set([data.y.item() for data in dataset]))

# Initialize the model
num_nodes = 12 * num_patches  # Number of nodes based on the patch size
model = GCN_25(dataset, num_nodes, num_patches, num_classes=num_classes)
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
        torch.save(checkpoint, os.path.join(save_dir, f'best_modelre{run_tag}_p{num_patches}_b{batch_size}_lr{lr}_e{epochs}.pt'))            

    #Testing the model on validation set:
    test_loss, test_accuracy, y_pred, y_test, y_prob = test_model(model, test_loader)
    # Convert to NumPy arrays
    y_true = np.array(y_test)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Save results
    save_dir_test = os.path.join(save_dir, f'test_results{run_tag}')
    os.makedirs(save_dir_test, exist_ok=True)
    np.save(os.path.join(save_dir_test, "y_true.npy"), y_true)
    np.save(os.path.join(save_dir_test, "y_pred.npy"), y_pred)
    np.save(os.path.join(save_dir_test, "y_prob.npy"), y_prob)




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
        model = GCN_25(dataset, num_nodes, num_patches, num_classes=num_classes).to(device)        
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
                torch.save(checkpoint, os.path.join(save_dir,f'best_model{run_tag}_fold{fold+1}_p{num_patches}_b{batch_size}_lr{lr}.pt'))
            
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
        
        