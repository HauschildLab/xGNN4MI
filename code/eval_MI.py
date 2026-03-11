"""
Evaluate a trained MI classification model.

This script loads saved predictions from the test set and computes evaluation
metrics such as accuracy, F1 score, MCC, ROC-AUC, and confusion matrices.

The script expects prediction files (y_true.npy, y_pred.npy, y_prob.npy)
to already exist in the results directory.

Environment variables:
    DATASET_LOCATION : path to the PTB-XL dataset directory
    SAVE_LOCATION    : directory where model outputs and results are stored
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import gc
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, roc_auc_score
from torch_geometric.loader import DataLoader
from ecg_to_graph_patch_MI import GraphDataset
import sys
import pandas as pd
import ast
from torch_geometric import seed_everything
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils.train_test import train_model, test_model
from utils.filter_scp import filtering_scp
from utils.xai_wrapper import GCNWrapper
from collections import Counter

from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.explain.metric import fidelity,groundtruth_metrics

num_patches = 25
batch_size = 32
lr = 0.001
epochs = 150

from GNN_patch25_MI import GCN_25

data_dir = os.environ.get('DATASET_LOCATION')
save_root = os.environ.get('SAVE_LOCATION', '.')
save_dir = os.path.join(save_root, 'MI_res')

# Get data
mi_labels = ["IMI", "ASMI", "NORM"]
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

##### Without Norm
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

print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Save results to file
save_dir_test = os.path.join(save_dir, 'test_results')
os.makedirs(save_dir_test, exist_ok=True)

# # Evaluate model
# model.eval()
# test_loss, test_accuracy, y_pred, y_test, y_prob = test_model(model, test_loader)

# # Convert to NumPy arrays
# y_true = np.array(y_test)
# y_pred = np.array(y_pred)
# y_prob = np.array(y_prob)

#np.save(os.path.join(save_dir_test, "y_true.npy"), y_true)
#np.save(os.path.join(save_dir_test, "y_pred.npy"), y_pred)
#np.save(os.path.join(save_dir_test, "y_prob.npy"), y_prob)


y_true = np.load(os.path.join(save_dir_test, "y_true.npy"))
y_pred = np.load(os.path.join(save_dir_test, "y_pred.npy"))
y_prob = np.load(os.path.join(save_dir_test, "y_prob.npy"))


# AUC
auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr')
print(f"Multi-Class AUC: {auc_score:.4f}")

# Confusion matrix + labels
class_names = [label_decoder[i] for i in range(len(label_decoder))]
conf_matrix = confusion_matrix(y_true, y_pred)

# F1 and MCC
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Macro F1 Score:     {f1_macro:.4f}")
print(f"Weighted F1 Score:  {f1_weighted:.4f}")

# Confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (MI Subtypes)")
#plt.savefig(os.path.join(save_dir_test, "confusion_matrix_mi_less.png"))
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Normalize the matrix for coloring (row-wise)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Prepare a figure
plt.figure(figsize=(10, 8))

# Plot normalized values for colors, but show absolute counts as annotations
sns.heatmap(conf_matrix_normalized, 
            annot=conf_matrix,
            fmt="d", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names,
            annot_kws={"size": 16},
            cbar_kws={"ticks": [0.2, 0.4, 0.6, 0.8]})

plt.xlabel("Predicted Label", fontsize=16)
plt.ylabel("True Label", fontsize=16)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.title("Confusion Matrix (MI Subtypes)", fontsize=16)
#plt.savefig(os.path.join(save_dir_test, "confusion_matrix_mi_less.svg"))
plt.show()


# True Positive Analysis
true_positive_mask = (y_pred == y_true)
tp_indices = np.where(true_positive_mask)[0]
tp_patient_ids = [test_dataset[i].ecg_id for i in tp_indices]

# Save TP info
#np.save(os.path.join(save_dir_test, "tp_indices.npy"), tp_indices)
#np.save(os.path.join(save_dir_test, "tp_patient_ids.npy"), tp_patient_ids)

print(f"Extracted {len(tp_indices)} true positive samples for further analysis.")


# wrongly classified indices
wrongly_classified_mask = (y_pred != y_true)
wrongly_classified_indices = np.where(wrongly_classified_mask)[0]
wrongly_classified_patient_ids = [test_dataset[i].ecg_id for i in wrongly_classified_indices]

# save wrongly classified info
#np.save(os.path.join(save_dir_test, "wrongly_classified_indices.npy"), wrongly_classified_indices)
#np.save(os.path.join(save_dir_test, "wrongly_classified_patient_ids.npy"), wrongly_classified_patient_ids)

#check, whether true positives and wrongly classified add up to all test samples
assert len(tp_indices) + len(wrongly_classified_indices) == len(test_dataset)
print(f"Extracted {len(wrongly_classified_indices)} wrongly classified samples for further analysis.")









###############################################################################################
###########################################################################################################
# Plot 10 wrongly classified for Kristin

import wfdb
import ecg_plot

# Find the numeric indices for ASMI and IMI
asmi_idx = [k for k, v in label_decoder.items() if v == "ASMI"][0]
imi_idx  = [k for k, v in label_decoder.items() if v == "IMI"][0]

# Mask for "true ASMI, predicted IMI"
asmi_pred_imi_mask = (y_true == asmi_idx) & (y_pred == imi_idx)
asmi_pred_imi_indices = np.where(asmi_pred_imi_mask)[0]

print(f"Found {len(asmi_pred_imi_indices)} ASMI cases predicted as IMI.")


# Take up to 10 of them
n_to_plot = min(10, len(asmi_pred_imi_indices))
selected_indices = asmi_pred_imi_indices[:n_to_plot]

print(f"Will plot {n_to_plot} ECGs (ASMI -> IMI).")


# Directory to save the plots (so you can print them)
plot_save_dir = os.path.join(save_dir, "ASMI_pred_IMI_plots")
os.makedirs(plot_save_dir, exist_ok=True)

selected_ecg_ids = []  # for logging / saving

for counter, idx in enumerate(selected_indices, start=1):
    data = test_dataset[idx]
    ecg_id = int(data.ecg_id.item())
    selected_ecg_ids.append(ecg_id)

    # Get row for this ECG
    if ecg_id not in df_new.index:
        print(f"WARNING: ECG ID {ecg_id} not found in df_new. Skipping.")
        continue

    row = df_new.loc[ecg_id]

    # ---- use HIGH-RES recording (500 Hz) instead of filename_lr ----
    record_path = os.path.join(path, row["filename_hr"])

    print(f"[{counter}/{n_to_plot}] Loading ECG ID {ecg_id} from {record_path}")

    try:
        # Load WFDB record (high-res, 500 Hz)
        record = wfdb.rdrecord(record_path)

        # wfdb gives shape (n_samples, 12); ecg_plot wants (12, n_samples)
        ecg = record.p_signal.T   # <-- transpose is important!

        title = f"ECG ID {ecg_id} | True: ASMI, Pred: IMI"

        # Plot standard 12-lead 10 s ECG (now really high-res)
        ecg_plot.plot_12(ecg, sample_rate=500, title=title)

        # Save as PNG for printing
        filename_base = f"ECG_{counter}_ECGID{ecg_id}_ASMI_pred_IMI"
        #ecg_plot.save_as_png(filename_base, plot_save_dir)


    except Exception as e:
        print(f"ERROR while processing ECG ID {ecg_id}: {e}")
        continue



# Mask for "true ASMI, predicted ASMI"
asmi_tp_mask = (y_true == asmi_idx) & (y_pred == asmi_idx)
asmi_tp_indices = np.where(asmi_tp_mask)[0]
imi_tp_mask = (y_true == imi_idx) & (y_pred == imi_idx)
imi_tp_indices = np.where(imi_tp_mask)[0]

print(f"Found {len(asmi_tp_indices)} correctly classified ASMI cases (ASMI -> ASMI).")
print(f"Found {len(imi_tp_indices)} correctly classified IMI cases (IMI -> IMI).")

# Take up to 5 of them
n_to_plot_tp = min(5, len(asmi_tp_indices))
selected_tp_indices = asmi_tp_indices[:n_to_plot_tp]

n_to_plot_tp_imi = min(5, len(imi_tp_indices))
selected_tp_indices_imi = imi_tp_indices[:n_to_plot_tp_imi]

print(f"Will plot {n_to_plot_tp} ECGs (ASMI -> ASMI, true positives).")
print(f"Will plot {n_to_plot_tp_imi} ECGs (IMI -> IMI, true positives).")

# Directory to save the TP ASMI plots
plot_save_dir_tp = os.path.join(save_dir, "ASMI_true_ASMI_plots")
os.makedirs(plot_save_dir_tp, exist_ok=True)
plot_save_dir_tp_imi = os.path.join(save_dir, "IMI_true_IMI_plots")
os.makedirs(plot_save_dir_tp_imi, exist_ok=True)

selected_ecg_ids_tp = []  # for logging / saving

for counter, idx in enumerate(selected_tp_indices, start=1):
    data = test_dataset[idx]
    ecg_id = int(data.ecg_id.item())
    selected_ecg_ids_tp.append(ecg_id)

    # Get row for this ECG
    if ecg_id not in df_new.index:
        print(f"WARNING (TP ASMI): ECG ID {ecg_id} not found in df_new. Skipping.")
        continue

    row = df_new.loc[ecg_id]

    # Use HIGH-RES recording (500 Hz)
    record_path = os.path.join(path, row["filename_hr"])

    print(f"[TP {counter}/{n_to_plot_tp}] Loading ECG ID {ecg_id} from {record_path}")

    try:
        # Load WFDB record (high-res, 500 Hz)
        record = wfdb.rdrecord(record_path)

        # wfdb gives shape (n_samples, 12); ecg_plot wants (12, n_samples)
        ecg = record.p_signal.T

        title = f"ECG ID {ecg_id} | True: ASMI, Pred: ASMI"

        # Plot standard 12-lead 10 s ECG (high-res)
        ecg_plot.plot_12(ecg, sample_rate=500, title=title)

        # Save as PNG for printing
        filename_base = f"TP_ECG_{counter}_ECGID{ecg_id}_ASMI_true_ASMI"
        #ecg_plot.save_as_png(filename_base, plot_save_dir_tp)

    except Exception as e:
        print(f"ERROR while processing TP ASMI ECG ID {ecg_id}: {e}")
        continue

# Optional: save list of TP ASMI ECG IDs
# np.save(
#     os.path.join(plot_save_dir_tp, "selected_ecg_ids_ASMI_true_ASMI.npy"),
#     np.array(selected_ecg_ids_tp)
# )

print(f"Saved up to {n_to_plot_tp} TP ASMI ECG plots to: {plot_save_dir_tp}")
