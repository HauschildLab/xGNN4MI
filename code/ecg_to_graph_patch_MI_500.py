import os
import numpy as np
import networkx as nx
import pandas as pd
import ast
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx
from scipy.sparse.csgraph import connected_components
#import ecg_plot
from sklearn.preprocessing import LabelEncoder
import os.path as osp

from helper_patch_500 import *


class GraphDataset(Dataset):
    def __init__(self, root, num_patches = 10,mi_labels=None, transform=None, pre_transform=None, pre_filter=None):
        self.records = find_records(root)
        self.num_patches = num_patches
        self.target_diseases = mi_labels or ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "PMI"]
        self.label_encoder = LabelEncoder()
        self.processed_files = []
        super().__init__(root, transform, pre_transform, pre_filter)        

    @property
    def raw_file_names(self):
        return self.records

    @property
    def processed_file_names(self):
        return self.processed_files
    
    @property 
    def processed_dir(self):
        return osp.join(self.root, f'processed_patch{self.num_patches}_MI_Norm_500')

    
    def process(self):
        self.processed_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.pt') and f.startswith('graph')]
        if self.processed_files:
            print(f"Loaded {len(self.processed_files)} processed files from {self.processed_dir}")
            return

        idx = 0
        data_list = []
        df = pd.read_csv(os.path.join(self.root, 'ptbxl_database.csv'), index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # STEP 1: Select MI subtypes only
        def extract_disease_label(scp_dict, allowed_diseases):
            matches = [code for code in scp_dict.keys() if code in allowed_diseases]
            return matches[0] if matches else None

        df['disease_label'] = df['scp_codes'].apply(lambda x: extract_disease_label(x, self.target_diseases))
        df = df[df['disease_label'].notnull()]
        df = df[df['validated_by_human'] == 1]
        df['disease_label_encoded'] = self.label_encoder.fit_transform(df['disease_label'])

        valid_ecg_ids = set(df.index)

        for record in self.raw_file_names:
            ecg_id = int(os.path.basename(record).split('_')[0])
            if ecg_id not in valid_ecg_ids:
                continue

            graph = create_graph_from_sample(os.path.join(self.root, record), self.num_patches)
            if graph is None:
                continue

            row = df.loc[ecg_id]
            encoded_label = row['disease_label_encoded']

            data = self.graph_to_data(graph, encoded_label, ecg_id)
            if self.pre_filter is not None:
                data = self.pre_filter(data)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'graph_data_{idx}.pt'))
            self.processed_files.append(osp.join(self.processed_dir, f'graph_data_{idx}.pt'))
            idx += 1


            

    def graph_to_data(self, graph, encoded_label, ecg_id):
        #print("Converting graph to data...")
        node_id_mapping = {node_name: i for i, node_name in enumerate(graph.nodes())}
        #print(f'node_id_mapping: {node_id_mapping}')
        x = []  # Node features (ECG signals)
        edge_index = []  # Edge connectivity
        feature_names = list(graph.nodes()) #store node names in correct order

        # Extract node features and edge connectivity from the graph
        for node in graph.nodes():
            if node:
                x.append(graph.nodes[node]['signal'])  # Node feature (ECG signal)
                node_id = node_id_mapping[node]
                #print(f'node_id: {node_id}')
                for neighbor in graph.neighbors(node):
                    #print(f'neighbor node {neighbor}')
                    if neighbor:
                        #print(f'Node1 {node} and neighbor node {neighbor}')
                        neighbor_id = node_id_mapping[neighbor]
                        #print(f'neighbor_id: {neighbor_id}')
                        edge_index.append([node_id, neighbor_id])
        
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        y = torch.tensor([encoded_label], dtype=torch.long)
        ecg_id = torch.tensor([ecg_id], dtype=torch.long)   

        # Construct PyG Data object
        data = Data(x=x, edge_index=edge_index, y = y,ecg_id=ecg_id, node_name = feature_names)
        
        return data
    
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'graph_data_{idx}.pt'),weights_only=False)
        return data
        
        
    #Get Labels back from decoded labels:
    def decode_label(self, label):
        return self.label_encoder.inverse_transform(label)
    
    
    
