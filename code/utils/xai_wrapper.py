"""
Wrapper class for adapting the GNN model to the PyTorch Geometric
explainability interface.

The wrapper converts the explainer inputs (x, edge_index, batch)
into a PyG Data object before forwarding them through the model.
"""

import torch


class GCNWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GCNWrapper, self).__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        # Import the Data object from torch_geometric
        from torch_geometric.data import Data

        # Extract batch if provided
        batch = kwargs.get('batch', None)

        # Create a Data object to pass to the original model
        data = Data(x=x, edge_index=edge_index, batch=batch)

        # Call the original model's forward method
        return self.model(data)