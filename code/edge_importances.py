"""
Compute edge importance scores for diagnostic superclass classification.

This script aggregates edge importance scores obtained from GNNExplainer
to identify the most influential spatial and temporal connections in
the ECG graph representation.

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
#import pyvista as pv
from collections import defaultdict, Counter
from utils.train_test import train_model, test_model
from utils.filter_scp import filtering_scp
from utils.xai_wrapper import GCNWrapper


from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.explain.metric import fidelity,groundtruth_metrics

# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs')
# parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
# parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
# parser.add_argument('-p', '--num_patches', type=int, default=10, help='Number of patches')

# args = parser.parse_args()

# epochs = int(args.epochs)
# batch_size = int(args.batch_size)
# lr = float(args.learning_rate)
# num_patches = int(args.num_patches)

num_patches = 25
batch_size = 32
lr = 0.001
epochs = 150

from GNN_patch25 import GCN_25

# Get data
data_dir = os.environ.get('DATASET_LOCATION')
save_dir = os.environ.get('SAVE_LOCATION', '.')
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



wrapped_model = GCNWrapper(model)
#Implement Explainer
explainer = Explainer(model=wrapped_model, algorithm=GNNExplainer(epochs=200), explanation_type='model', model_config=dict(mode = 'multiclass_classification', task_level = 'graph', return_type = 'log_probs', ), node_mask_type=None, edge_mask_type='object')


# Select one random TP patient
#selected_patient_idx = np.random.choice(tp_indices)
selected_patient_idx=1247
#selected_patient_idx = 1634
print(f"Selected True Positive Patient Index: {selected_patient_idx}")

# Fetch the patient's data
data = test_dataset[selected_patient_idx].to(device)

# Get feature importance using the explainer
explanation = explainer(data.x, data.edge_index)

explanation.edge_mask


# Load true positive indices
save_dir_test = os.path.join(save_dir, 'test_results')

tp_indices = np.load(os.path.join(save_dir_test,'tp_indices.npy'))
tp_patient_ids = np.load(os.path.join(save_dir_test,'tp_patient_ids.npy'))


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

#torch.save(normalized_edge_importance, os.path.join(save_dir_test,"normalized_edge_importance.pth"))

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
top_k = 10

# Initialize dictionaries to store edge importance scores per disease
edge_importance_sum_per_disease = defaultdict(lambda: defaultdict(float))  # {disease: {edge: importance_sum}}
edge_count_per_disease = defaultdict(lambda: defaultdict(int))  # {disease: {edge: count}}

# Process each TP patient
for i in tp_indices:
    data = test_dataset[i].to(device)

    # Get disease labels for this patient
    patient_diseases = df_new.loc[int(test_dataset[i].ecg_id.item()), "scp_codes"].keys()

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

    if i % 100 == 0:
        print(f"Processed {i} samples...")

# Normalize by occurrence count (optional, to get an average importance per edge)
normalized_edge_importance_per_disease = {
    disease: {edge: edge_importance_sum_per_disease[disease][edge] / edge_count_per_disease[disease][edge]
              for edge in edge_importance_sum_per_disease[disease]}
    for disease in edge_importance_sum_per_disease
}

#torch.save(normalized_edge_importance_per_disease, os.path.join(save_dir_test,"normalized_edge_importance_per_disease.pth"))

print("Saved normalized edge importance per disease to 'normalized_edge_importance_per_disease.pth'.")

#normalized_edge_importance_per_disease = torch.load(os.path.join(save_dir_test,"normalized_edge_importance_per_disease.pth"))

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
#mean_edge_df.to_csv(os.path.join(save_dir_test,"mean_edge_importance_node_names.csv"), index=False)

 
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
#type_df.to_csv(os.path.join(save_dir_test,"edge_type_percentages_per_disease.csv"), index=False)

print("Saved edge type percentages per disease to: edge_type_percentages_per_disease.csv")






######With diagnosis:
# Initialize dictionaries for edge importances per diagnosis
edge_importance_sum_per_diagnosis = defaultdict(lambda: defaultdict(float))  # {diagnosis: {edge: importance_sum}}
edge_count_per_diagnosis = defaultdict(lambda: defaultdict(int))  # {diagnosis: {edge: count}}

# Diagnosis mapping from numeric label to name
diagnosis_mapping = {0: "CD", 1: "HYP", 2: "MI", 3: "NORM", 4: "STTC"}

# Process each TP patient
for i in tp_indices:
    data = test_dataset[i].to(device)

    # Get diagnosis class from label
    diag_label = test_dataset[i].y.item()
    diagnosis = diagnosis_mapping[diag_label]

    # Generate explanation
    explanation = explainer(data.x, data.edge_index)

    # Get edge importance scores
    edge_importance = explanation.edge_mask.cpu().numpy()

    # Aggregate edge importances per diagnosis
    for j, score in enumerate(edge_importance):
        edge_tuple = tuple(data.edge_index[:, j].cpu().numpy())  # Convert edge to tuple (source, target)

        edge_importance_sum_per_diagnosis[diagnosis][edge_tuple] += score
        edge_count_per_diagnosis[diagnosis][edge_tuple] += 1  # Count occurrences

    if i % 100 == 0:
        print(f"Processed {i} samples...")

# Normalize edge importance per diagnosis
normalized_edge_importance_per_diagnosis = {
    diagnosis: {
        edge: edge_importance_sum_per_diagnosis[diagnosis][edge] / edge_count_per_diagnosis[diagnosis][edge]
        for edge in edge_importance_sum_per_diagnosis[diagnosis]
    }
    for diagnosis in edge_importance_sum_per_diagnosis
}

# Save to .pth file
#torch.save(normalized_edge_importance_per_diagnosis, os.path.join(save_dir_test, "normalized_edge_importance_per_diagnosis.pth"))
print("Saved normalized edge importance per diagnosis to 'normalized_edge_importance_per_diagnosis.pth'.")

# Load the diagnosis-level edge importance file
#normalized_edge_importance_per_diagnosis = torch.load(os.path.join(save_dir_test,"normalized_edge_importance_per_diagnosis.pth"))

records = []

# Calculate mean edge importance scores for each lead pair across all diagnoses
for diagnosis, edges in normalized_edge_importance_per_diagnosis.items():
    print(f"\nProcessing diagnosis: {diagnosis}...")

    lead_pair_stats = defaultdict(lambda: [0.0, 0])  # {("I", "II"): [sum, count]}

    for (src, dst), score in edges.items():
        lead1 = data.node_name[src].split("_")[0]
        lead2 = data.node_name[dst].split("_")[0]

        if lead1 == lead2:
            continue  # skip self-lead edges

        pair = tuple(sorted((lead1, lead2)))
        lead_pair_stats[pair][0] += score
        lead_pair_stats[pair][1] += 1

    for (lead1, lead2), (total, count) in lead_pair_stats.items():
        mean_score = total / count
        records.append({
            "diagnosis": diagnosis,
            "lead_1": lead1,
            "lead_2": lead2,
            "mean_importance_score": mean_score
        })

# Save to CSV
mean_edge_df_diag = pd.DataFrame(records)
#mean_edge_df_diag.to_csv(os.path.join(save_dir_test,"mean_edge_importance_node_names_diagnosis.csv"), index=False)
print("Saved mean edge importance scores per lead pair and diagnosis to CSV.")

  