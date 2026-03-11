import numpy as np
import os
import sys
import networkx as nx
import scipy.signal as sig

def find_records(folder):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extensions = os.path.splitext(file)[1]
            if extensions == '.hea':
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                if 'records100' not in record: 
                    records.add(record)
    records = sorted(records)
    return records


def get_header_file(record):
    if not record.endswith('.hea'):
        header_file = record + '.hea'
    else:
        header_file = record
    return header_file

def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string

def get_signal_files_from_header(string):
    signal_files = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i==0 and not l.startswith('#'):
            num_channels = int(arrs[1])
        elif i<=num_channels and not l.startswith('#'):
            signal_file = arrs[0]
            if signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
    return signal_files

def get_signal_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    signal_files = get_signal_files_from_header(header)
    return signal_files
    

def load_signal(record):
    import wfdb
    
    signal_files = get_signal_files(record)
    if signal_files:
        signal, fields = wfdb.rdsamp(record)

    
    else:
        signal, fields = None, None
    return signal, fields

# Load the signal(s) for a record.
def load_signals(record):
    return load_signal(record)


def create_graph_from_sample(record, num_patches=10):
    signal, fields = load_signal(record)
    if signal is not None:
        limb_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
        chest_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        specific_leads = ['I', 'AVF', 'V4', 'V5']
    
        # Create empty graph
        G = nx.Graph()
    
        # Add nodes with patch data
        sig_names = fields['sig_name']
        for i, lead in enumerate(sig_names):
            signal_data = signal[:, i]
            patch_size = len(signal_data) // num_patches
            for j in range(num_patches):
                patch_data = signal_data[j*patch_size:(j+1)*patch_size]
                G.add_node(f"{lead}_p{j}", signal=patch_data)
            
        # Add temporal edges (within same lead)
        for i, lead in enumerate(sig_names):
            num_patches = len(signal[:, i]) // patch_size
            for j in range(num_patches - 1):
                G.add_edge(f"{lead}_p{j}", f"{lead}_p{j+1}")
        
        # Add spatial edges (within limb and chest leads)
        for lead1 in limb_leads:
            for lead2 in limb_leads:
                if lead1 != lead2:
                    num_patches = len(signal[:, sig_names.index(lead1)]) // patch_size
                    for j in range(num_patches):
                        G.add_edge(f"{lead1}_p{j}", f"{lead2}_p{j}")
        
        for lead1 in chest_leads:
            for lead2 in chest_leads:
                if lead1 != lead2:
                    num_patches = len(signal[:, sig_names.index(lead1)]) // patch_size
                    for j in range(num_patches):
                        G.add_edge(f"{lead1}_p{j}", f"{lead2}_p{j}")
        
        # Add specific inter-lead connections
        for lead1 in specific_leads:
            for lead2 in specific_leads:
                if lead1 != lead2:
                    num_patches = len(signal[:, sig_names.index(lead1)]) // patch_size
                    for j in range(num_patches):
                        G.add_edge(f"{lead1}_p{j}", f"{lead2}_p{j}")
                    
        return G
    
    else:
        return None


    
