"""
Compute node importance scores for MI classification using GNNExplainer.

The script explains predictions for true-positive test samples and aggregates
node importance scores across patients. Results are saved as CSV files
for further analysis.

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
from ecg_to_graph_patch_MI import GraphDataset
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

from GNN_patch25_MI import GCN_25


# Get data
mi_labels = ["IMI", "ASMI", "NORM"]
data_dir = os.environ.get('DATASET_LOCATION')
save_root = os.environ.get('SAVE_LOCATION', '.')
save_dir = os.path.join(save_root, 'MI_res')
#print(data_dir)
path = os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
dataset = GraphDataset(root=path, num_patches=num_patches, mi_labels=mi_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_classes = len(set([data.y.item() for data in dataset]))

num_nodes = 12 * num_patches
model = GCN_25(dataset, num_nodes, num_patches, num_classes)


    
#load the best model (saved in the same directory)
state_dict = torch.load(os.path.join(save_dir,f'best_modelre_p{num_patches}_b{batch_size}_lr{lr}_e{epochs}.pt'),map_location=torch.device('cpu') )

model.load_state_dict(state_dict['model'])
print('Model loaded')

model = model.to(device)

# Split the dataset into training and testing sets
seed_everything(42)
# Load ptbxl_database.csv and parse scp_codes
df = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

# Extract one MI disease label per sample
def extract_disease_label(scp_dict, allowed_labels):
    matches = [code for code in scp_dict.keys() if code in allowed_labels]
    return matches[0] if matches else None


###########Without Norm:
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

# Encode labels
label_encoder = {label: idx for idx, label in enumerate(sorted(df_new['disease_label'].unique()))}
label_decoder = {v: k for k, v in label_encoder.items()}


# Stratified fold split
train_ids = df_new[df_new.strat_fold <= 8].index.values
val_ids   = df_new[df_new.strat_fold == 9].index.values
test_ids  = df_new[df_new.strat_fold == 10].index.values

# Filter the dataset (by ECG ID in graph data)
train_dataset = [data for data in dataset if data.ecg_id.item() in train_ids]
val_dataset   = [data for data in dataset if data.ecg_id.item() in val_ids]
test_dataset  = [data for data in dataset if data.ecg_id.item() in test_ids]


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


###############################################################################################################
###############################################################################################################
###############################################################################################################
#For all TP patients
###############################################################################################################
###############################################################################################################
###############################################################################################################
top_k = 10

#load true positive indices
save_dir_test = os.path.join(save_dir, 'test_results')
os.makedirs(save_dir_test, exist_ok=True)
tp_indices = np.load(os.path.join(save_dir_test, "tp_indices.npy"))

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
importance_df.to_csv(os.path.join(save_dir_test,"node_importance_scores_tp_mi.csv"))

# Load the existing importance scores CSV
importance_df = pd.read_csv(os.path.join(save_dir_test,"node_importance_scores_tp_mi.csv"), index_col="Node Index")

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
importance_df = pd.read_csv(os.path.join(save_dir_test,"node_importance_scores_tp_mi.csv"), index_col="Node Index")

# Load the TP patient indices (which were used for computing importance scores)
tp_indices = np.load(os.path.join(save_dir_test,"tp_indices.npy"))

# Get true diagnoses for these TP patients
diagnoses = [test_dataset[i].y.item() for i in tp_indices]  # Extracting labels for TP patients

# Map numerical labels to class names
diagnoses = [label_decoder[d] for d in diagnoses]  # Convert numeric labels to class names

#Define MI subtype list 
mi_labels = ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "PMI", "NORM"]

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

importance_df.to_csv(os.path.join(save_dir_test,"node_importance_scores_tp_mi_disease.csv"))
