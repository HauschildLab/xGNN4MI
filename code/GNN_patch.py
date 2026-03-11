import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, dataset, num_nodes, num_patches):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_patches = num_patches
        
        # Adjust the FFN to match the input size for the first GCN layer
        self.ffn1 = nn.Sequential(
            nn.Linear(100,3840),  # Adjust the output size to match the input size for GCNConv
            nn.BatchNorm1d(3840),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(3840, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.5)
        )
        
        self.conv1 = GCNConv(dataset.num_features, 64)  # Adjusted input size to match FFN output
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 8)
        self.conv5 = GCNConv(8, 4)
    
        
        # Calculate the correct input size for the fully connected layer
        self.fc = nn.Linear((self.num_nodes * 4), dataset.num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()

        #print(f'Input shape: {x.shape}')
        x = self.ffn1(x)  # FFN
        #print(f'After FFN shape: {x.shape}')
        
        x = self.conv1(x, edge_index)
        #print(f'After conv1 shape: {x.shape}')
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        #print(f'After conv2 shape: {x.shape}')
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        #print(f'After conv3 shape: {x.shape}')
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        #print(f'After conv4 shape: {x.shape}')
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)
        #print(f'After conv5 shape: {x.shape}')
        x = F.relu(x)
        x = F.dropout(x, training=self.training)


        # Reshape the tensor to match the input size for the fully connected layer
        x = torch.reshape(x, (-1,(self.num_nodes * 4)))
        #print(f'After reshape shape: {x.shape}')

        x = self.fc(x)
        #print(f'Output shape: {x.shape}')

        return x
