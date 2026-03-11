"""
Compute node importance scores for diagnostic superclass classification.

This script applies GNNExplainer to true-positive test samples and aggregates
node importance scores across patients. Results are saved as CSV files
for downstream analysis and visualization.

Environment variables:
    DATASET_LOCATION : path to the PTB-XL dataset directory
    SAVE_LOCATION    : directory where experiment results are stored
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import gc
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader
from ecg_to_graph_patch import GraphDataset
import sys
import pandas as pd
import ast
from torch_geometric import seed_everything
import argparse
# import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.utils import to_networkx
# import pyvista as pv
from collections import defaultdict, Counter
from utils.train_test import train_model, test_model
from utils.filter_scp import filtering_scp
from utils.xai_wrapper import GCNWrapper


from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.explain.metric import fidelity,groundtruth_metrics


num_patches = 25
batch_size = 32
lr = 0.001
epochs = 150

from GNN_patch25 import GCN_25

# Get data
data_dir = os.environ.get('DATASET_LOCATION')
save_dir = os.environ.get('SAVE_LOCATION', '.')
#print(data_dir)
path = os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
dataset = GraphDataset(root=path, num_patches=num_patches)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_nodes = 12 * num_patches
model = GCN_25(dataset, num_nodes, num_patches)
    
#load the best model (saved in the same directory)
state_dict = torch.load(os.path.join(save_dir, f'best_model_p{num_patches}_b{batch_size}_lr{lr}_e{epochs}.pt'),map_location=torch.device('cpu') )
model.load_state_dict(state_dict['model'])
print('Model loaded')

model = model.to(device)

# Split the dataset into training and testing sets
seed_everything(42)
num_samples = len(dataset)

df_new = filtering_scp(path = path)

train_mask = df_new.strat_fold <= 8
val_mask = df_new.strat_fold == 9
test_mask = df_new.strat_fold == 10

train_dataset =dataset[np.array(train_mask)]
val_dataset = dataset[np.array(val_mask)]
test_dataset = dataset[np.array(test_mask)]

# Create DataLoader instances for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


###############################################################################################################
###############################################################################################################
###############################################################################################################
#For one patient
###############################################################################################################
###############################################################################################################
###############################################################################################################
data = dataset[0]
data = data.to(device)

out = model(data)
pred = out.argmax(dim=1)

wrapped_model = GCNWrapper(model)
#Implement Explainer
explainer = Explainer(model=wrapped_model, algorithm=GNNExplainer(epochs=200), explanation_type='model', model_config=dict(mode = 'multiclass_classification', task_level = 'graph', return_type = 'log_probs', ), node_mask_type='object', edge_mask_type='object')

explanation = explainer(data.x, data.edge_index)
# Extract node importance scores
node_importance = explanation.node_mask.cpu().numpy()

# Sort nodes by importance
flat_node_importance = node_importance.flatten()

# Number of top important nodes you want to see
top_k = 10

# Get indices of top_k nodes
top_k_indices = np.argsort(flat_node_importance)[-top_k:]

# Get the importance scores of the top_k nodes
top_k_scores = flat_node_importance[top_k_indices]

print("Top {} important nodes:".format(top_k))
for idx, score in zip(top_k_indices, top_k_scores):
    print(f"Node index: {idx}, Importance score: {score}")




###############################################################################################################
###############################################################################################################
###############################################################################################################
#For all TP patients
###############################################################################################################
###############################################################################################################
###############################################################################################################

save_dir_test = os.path.join(save_dir, 'test_results')
os.makedirs(save_dir_test, exist_ok=True)

#load true positive indices
tp_indices = np.load(os.path.join(save_dir_test,'tp_indices.npy'))
tp_patient_ids = np.load(os.path.join(save_dir_test,'tp_patient_ids.npy'))
feature_counter = Counter()
total_samples = 0  # Track processed samples
total_node_importance = 0

# Loop over TP samples
for i in tp_indices:
    data = test_dataset[i].to(device)
    explanation = explainer(data.x, data.edge_index)
    node_importance = explanation.node_mask.cpu().numpy().flatten()  # Flatten to 1D array
    
    top_k_indices = np.argsort(node_importance)[-top_k:]

    # Aggregate node importances (sum over all samples)
    total_node_importance += node_importance  # Sum over all test instances
    for idx in top_k_indices.tolist():
        feature_counter[int(idx)] += 1
    
    
    total_samples += 1
    if i % 100 == 0:
        print(f"Processed {i} samples...")

print(f"Total samples processed: {total_samples}")
print("Most frequent nodes in top 10 most important node Indices:", feature_counter.most_common(10))

# Get indices of the top 10 most important nodes
top_k = 10
top_nodes = np.argsort(total_node_importance)[-top_k:]  # Highest importance nodes

# Get corresponding importance scores
top_scores = total_node_importance[top_nodes]

# Print results
print("Top 10 Most Important Nodes Over All TP Test Data:")
for idx, score in zip(top_nodes, top_scores):
    print(f"Node index: {idx}, Aggregated Importance score: {score}")
    
    

# Initialize a dictionary to store the importance scores of nodes for each patient.
node_importance_dict = defaultdict(list)
patient_indices = []
explainer = Explainer(model=wrapped_model, algorithm=GNNExplainer(epochs=200), explanation_type='model', model_config=dict(mode = 'multiclass_classification', task_level = 'graph', return_type = 'log_probs', ), node_mask_type='object', edge_mask_type='object')
total_samples = 0  # Track processed samples
# Iterate over the test dataset and calculate the importance scores for each node.
for patient_idx, data in enumerate(tp_indices):
    data = test_dataset[data].to(device)
    explanation = explainer(data.x, data.edge_index)
    node_importance = explanation.node_mask.cpu().numpy().flatten()  # Flatten to 1D array

    node_importance_dict[patient_idx].extend(node_importance)
    patient_indices.append(patient_idx)
    
    total_samples += 1
    if patient_idx % 100 == 0:
        print(f"Processed {patient_idx} samples...")
    
# Convert the dictionary to a DataFrame
importance_df = pd.DataFrame(node_importance_dict)
importance_df.index.name = "Node Index"
importance_df.columns = [f"Patient_{idx}" for idx in patient_indices]

# Save the importance scores to a CSV file
importance_df.to_csv(os.path.join(save_dir_test,"node_importance_scores_tp.csv"))
print("Node importance scores saved to 'node_importance_scores_tp.csv'.")

# Load the existing importance scores CSV
importance_df = pd.read_csv(os.path.join(save_dir_test,"node_importance_scores_tp.csv"), index_col="Node Index")

# Define the correct lead mapping
leads = {
    "Lead I": list(range(0, 25)),
    "Lead II": list(range(25, 50)),
    "Lead III": list(range(50, 75)),
    "Lead aVR": list(range(75, 100)),
    "Lead aVL": list(range(100, 125)),
    "Lead aVF": list(range(125, 150)),
    "Lead V1": list(range(150, 175)),
    "Lead V2": list(range(175, 200)),
    "Lead V3": list(range(200, 225)),
    "Lead V4": list(range(225, 250)),
    "Lead V5": list(range(250, 275)),
    "Lead V6": list(range(275, 300)),
}

# Load the existing importance scores (computed earlier)
importance_df = pd.read_csv(os.path.join(save_dir_test,"node_importance_scores_tp.csv", index_col="Node Index"))

# Load the TP patient indices (which were used for computing importance scores)
tp_indices = np.load(os.path.join(save_dir_test,"tp_indices.npy"))

# Get true diagnoses for these TP patients
diagnoses = [test_dataset[i].y.item() for i in tp_indices]  # Extracting labels for TP patients

# Map numerical labels to class names
diagnosis_mapping = {0: "CD", 1: "HYP", 2: "MI", 3: "NORM", 4: "STTC"}
diagnoses = [diagnosis_mapping[d] for d in diagnoses]  # Convert numeric labels to class names

# Create a DataFrame mapping patients to their diagnosis
diagnosis_df = pd.DataFrame({"Patient": [f"Patient_{idx}" for idx in range(len(tp_indices))], "Diagnosis": diagnoses})
diagnosis_df.set_index("Patient", inplace=True)

# Merge the importance scores with diagnosis info
importance_df.columns = [f"Patient_{idx}" for idx in range(len(tp_indices))]  # Rename columns to match
importance_df = importance_df.T.join(diagnosis_df)  # Add diagnosis column


# Ensure df_new index is integer-based for proper matching
df_new.index = df_new.index.astype(int)

# Get all unique SCP codes (diseases)
unique_diseases = set()
for scp_dict in df_new.scp_codes:
    unique_diseases.update(scp_dict.keys())

# Sort diseases alphabetically
unique_diseases = sorted(unique_diseases)
print(f"Found {len(unique_diseases)} unique diseases.")

# Create mapping from ECG ID to diseases
ecg_to_disease = {int(ecg_id): list(scp_dict.keys()) for ecg_id, scp_dict in zip(df_new.index, df_new.scp_codes)}

# Extract ECG IDs for TP patients
tp_patient_ids = [int(test_dataset[i].ecg_id.item()) for i in tp_indices]  # Convert to real ECG IDs

# Get diseases for TP patients
tp_diseases = [", ".join(ecg_to_disease.get(pid, [])) for pid in tp_patient_ids]  # Join diseases as a string

# Create a DataFrame mapping patients to diseases
disease_df = pd.DataFrame({"Patient": [f"Patient_{idx}" for idx in range(len(tp_indices))], "Diseases": tp_diseases})
disease_df.set_index("Patient", inplace=True)

# Merge disease information into importance_df
importance_df = importance_df.join(disease_df)

# Print sample to verify
print(importance_df.head())

importance_df.to_csv(os.path.join(save_dir_test,"node_importance_scores_tp_diagnoses.csv"))



# Find TP patients that have IRBBB in their list of diagnoses
tp_irbbb_patient_ids = [pid for pid in tp_patient_ids if "IRBBB" in ecg_to_disease.get(pid, [])]

print(f"Total TP patients with IRBBB: {len(tp_irbbb_patient_ids)}")
#44

# Ensure df_new index is integer-based for matching
df_new.index = df_new.index.astype(int)

# Ensure tp_patient_ids are also in integer format
tp_patient_ids = [int(ecg_id) for ecg_id in tp_patient_ids]

# Check for missing TP patient IDs
missing_ecg_ids = [ecg_id for ecg_id in tp_patient_ids if ecg_id not in df_new.index]
if missing_ecg_ids:
    print(f"Warning: {len(missing_ecg_ids)} TP ECG IDs not found in df_new! First few: {missing_ecg_ids[:10]}")

# Create a mapping from ECG ID to diseases (extracting only the disease names)
ecg_to_disease = {ecg_id: ", ".join(scp_dict.keys()) for ecg_id, scp_dict in df_new["scp_codes"].items()}

# Assign diseases (SCP codes) to TP patients
tp_diseases = [ecg_to_disease.get(ecg_id, "Unknown") for ecg_id in tp_patient_ids]

# Convert to DataFrame
diseases_df = pd.DataFrame({"Patient": [f"Patient_{idx}" for idx in range(len(tp_indices))], "Diseases": tp_diseases})
diseases_df.set_index("Patient", inplace=True)

# If "Diseases" already exists, drop it first
if "Diseases" in importance_df.columns:
    importance_df = importance_df.drop(columns=["Diseases"])

# Now safely add the new "Diseases" column
importance_df = importance_df.join(diseases_df)

# Verify the changes
print(importance_df.head())  # Now importance_df has "Diagnosis" and "Diseases"
# Extract all unique diseases
all_diseases = sorted(set([d for disease_list in importance_df["Diseases"].dropna().str.split(", ") for d in disease_list]))

print(f"Total Unique Diseases: {len(all_diseases)}")
print("Sample Diseases:", all_diseases[:10])  # Print a sample


