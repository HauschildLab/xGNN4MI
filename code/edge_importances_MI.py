"""
Compute edge importance scores for MI classification using GNNExplainer.

The script analyzes explanations for true-positive test samples and aggregates
edge importance scores across patients to identify the most relevant
connections in the ECG graph representation.

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
#import pyvista as pv
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
save_root = os.environ.get('SAVE_LOCATION','.')
save_dir = os.path.join(save_root, 'MI_res')
#print(data_dir)
path = os.path.join(data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
dataset = GraphDataset(root=path, num_patches=num_patches, mi_labels=mi_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_nodes = 12 * num_patches
num_classes = len(set([data.y.item() for data in dataset]))
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

######### Without Norm
# df['disease_label'] = df['scp_codes'].apply(lambda x: extract_disease_label(x, mi_labels))
# df_new = df[df['disease_label'].notnull()]
# df_new = df_new[df_new['validated_by_human'] == 1]
######## Superclasses 
# #df_new['superclass'] = df_new['disease_label'].map(subclass_to_super)


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

wrapped_model = GCNWrapper(model)
#Implement Explainer
explainer = Explainer(model=wrapped_model, algorithm=GNNExplainer(epochs=200), explanation_type='model', model_config=dict(mode = 'multiclass_classification', task_level = 'graph', return_type = 'log_probs', ), node_mask_type=None, edge_mask_type='object')

# Load true positive indices
save_dir_test = os.path.join(save_dir, 'test_results')
tp_indices = np.load(os.path.join(save_dir_test, "tp_indices.npy"))

edge_importance_sum = defaultdict(float)  # Dictionary to store summed importance scores
edge_count = defaultdict(int)  # Track frequency of edges appearing in explanations

# Loop over TP samples
for i in tp_indices:
    data = test_dataset[i].to(device)
    
    # Generate explanation
    explanation = explainer(data.x, data.edge_index)
    
    # Get edge importance scores
    edge_importance = explanation.edge_mask.cpu().numpy()

    # Aggregate edge importances
    for j, score in enumerate(edge_importance):
        edge_tuple = tuple(data.edge_index[:, j].cpu().numpy())  # Convert edge to tuple (source, target)
        edge_importance_sum[edge_tuple] += score
        edge_count[edge_tuple] += 1  # Count occurrences of edge in explanations

    if i % 100 == 0:
        print(f"Processed {i} samples...")

# Normalize by occurrence count (optional, if you want an average importance per edge)
normalized_edge_importance = {edge: edge_importance_sum[edge] / edge_count[edge] 
                              for edge in edge_importance_sum}


# Get the top-K most important edges
top_k = 20  # Adjust based on how many important edges you want
most_important_edges = sorted(normalized_edge_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]


# Print results with node names
print(f"Top {top_k} Most Important Edges Over All TP Test Data:")
for edge, score in most_important_edges:
    node_1 = data.node_name[edge[0]]  # Convert index to node name
    node_2 = data.node_name[edge[1]]
    print(f"Edge ({node_1}, {node_2}), Aggregated Importance Score: {score}")
    
 
 
 # Define top-K most important edges to extract
top_k = 100

# Initialize dictionaries to store edge importance scores per disease
edge_importance_sum_per_disease = defaultdict(lambda: defaultdict(float))  # {disease: {edge: importance_sum}}
edge_count_per_disease = defaultdict(lambda: defaultdict(int))  # {disease: {edge: count}}

# Process each TP patient
for i in tp_indices:
    data = test_dataset[i].to(device)

    # Get disease labels for this patient
    patient_diseases = df_new.loc[int(test_dataset[i].ecg_id.item()), "scp_codes"].keys()
    # ecg_id = int(data.ecg_id.item())
    # superclass = df_new.loc[ecg_id, "superclass"]

    # Generate explanation
    explanation = explainer(data.x, data.edge_index)

    # Get edge importance scores
    edge_importance = explanation.edge_mask.cpu().numpy()

    # Aggregate edge importances per disease
    for j, score in enumerate(edge_importance):
        edge_tuple = tuple(data.edge_index[:, j].cpu().numpy())  # Convert edge to tuple (source, target)

        for disease in patient_diseases:
            edge_importance_sum_per_disease[disease][edge_tuple] += score
            edge_count_per_disease[disease][edge_tuple] += 1  # Count occurrences
            
        # edge_importance_sum_per_disease[superclass][edge_tuple] += score
        # edge_count_per_disease[superclass][edge_tuple] += 1  # Count occurrences

    if i % 100 == 0:
        print(f"Processed {i} samples...")

# Normalize by occurrence count (optional, to get an average importance per edge)
normalized_edge_importance_per_disease = {
    disease: {edge: edge_importance_sum_per_disease[disease][edge] / edge_count_per_disease[disease][edge]
              for edge in edge_importance_sum_per_disease[disease]}
    for disease in edge_importance_sum_per_disease
}

torch.save(normalized_edge_importance_per_disease, os.path.join(save_dir_test,"normalized_edge_importance_per_disease.pth"))
print("Saved normalized edge importance per disease to 'normalized_edge_importance_per_disease.pth'.")

normalized_edge_importance_per_disease = torch.load(os.path.join(save_dir_test,"normalized_edge_importance_per_disease.pth"))

data = test_dataset[0]
# Extract and print the top-K most important edges per disease
for disease, edges in normalized_edge_importance_per_disease.items():
    print(f"\nProcessing disease: {disease}...")

    # Sort edges by importance
    most_important_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Print results with node names
    print(f"Top {top_k} Most Important Edges for {disease}:")
    for edge, score in most_important_edges:
        node_1 = data.node_name[edge[0]]  # Convert index to node name
        node_2 = data.node_name[edge[1]]
        print(f"Edge ({node_1}, {node_2}), Aggregated Importance Score: {score:.6f}")   
 
 
 
 
#Calculate mean edge importance scores for each lead pair across all diseases/diagnosis
records = []

for disease, edges in normalized_edge_importance_per_disease.items():
    print(f"\nProcessing disease: {disease}...")

    # Use running average: {("I", "II"): [sum, count]}
    lead_pair_stats = defaultdict(lambda: [0.0, 0])

    for (src, dst), score in edges.items():
        lead1 = data.node_name[src].split("_")[0]
        lead2 = data.node_name[dst].split("_")[0]

        if lead1 == lead2:
            continue  # skip self-lead edges

        pair = tuple(sorted((lead1, lead2)))
        lead_pair_stats[pair][0] += score
        lead_pair_stats[pair][1] += 1

    # Store mean scores
    for (lead1, lead2), (total, count) in lead_pair_stats.items():
        mean_score = total / count
        records.append({
            "disease": disease,
            "lead_1": lead1,
            "lead_2": lead2,
            "mean_importance_score": mean_score
        })


# Save to CSV
mean_edge_df = pd.DataFrame(records)
mean_edge_df.to_csv(os.path.join(save_dir_test,"mean_edge_importance_node_names.csv"), index=False)
 
#Getting timely and spatial counts for each disease:

def classify_edge(node_1, node_2):
    lead_1, patch_1 = node_1.split("_p")
    lead_2, patch_2 = node_2.split("_p")
    
    if lead_1 == lead_2 and patch_1 != patch_2:
        return "timely"
    elif patch_1 == patch_2 and lead_1 != lead_2:
        return "spatial"
    else:
        return "other"  # just in case

results = []

for disease, edges in normalized_edge_importance_per_disease.items():
    # Sort edges by importance
    most_important_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:top_k]

    timely = 0
    spatial = 0
    other = 0

    for (i, j), score in most_important_edges:
        node_1 = data.node_name[i]
        node_2 = data.node_name[j]
        
        connection_type = classify_edge(node_1, node_2)
        if connection_type == "timely":
            timely += 1
        elif connection_type == "spatial":
            spatial += 1

    total = timely + spatial + other
    results.append({
        "disease": disease,
        "timely_count": timely,
        "spatial_count": spatial,
        "timely_percentage": timely / total * 100,
        "spatial_percentage": spatial / total * 100,
    })

# Convert to DataFrame
type_df = pd.DataFrame(results)

# Save to CSV
type_df.to_csv(os.path.join(save_dir_test,"edge_type_percentages_per_disease.csv"), index=False)

print("Saved edge type percentages per disease to: IEEE_res/MI_res/edge_type_percentages_per_disease.csv")
 