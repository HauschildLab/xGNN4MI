import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GCN_25(torch.nn.Module):
    def __init__(self, dataset, num_nodes, num_patches, num_classes):
        super(GCN_25, self).__init__()
        self.num_nodes = num_nodes
        self.num_patches = num_patches
        self.num_classes = num_classes
        
        # Adjust the FFN to match the input size for the first GCN layer
        self.ffn1 = nn.Sequential(
            nn.Linear(40, 9600),
            nn.BatchNorm1d(9600),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(9600, 40),
            nn.BatchNorm1d(40),
            nn.Dropout(p=0.5)
        )
        
        self.conv1 = GCNConv(dataset.num_features, 64)  # Adjusted input size to match FFN output
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 8)
        self.conv5 = GCNConv(8, 4)
    
        
        # Calculate the correct input size for the fully connected layer
        self.fc = nn.Linear((self.num_nodes * 4), self.num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()

        x = self.ffn1(x)  # FFN
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)


        # Reshape the tensor to match the input size for the fully connected layer
        x = torch.reshape(x, (-1,(self.num_nodes * 4)))

        x = self.fc(x)

        return x
