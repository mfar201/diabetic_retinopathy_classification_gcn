import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConstruction(nn.Module):
    def __init__(self, k=4, radius=0.1, feature_weight=0.5):
        super(GraphConstruction, self).__init__()
        self.k = k
        self.radius = radius
        self.feature_weight = feature_weight
        
    def create_edge_index(self, spatial_features, semantic_features):
        batch_size = spatial_features.size(0)
        device = spatial_features.device
        
        spatial_features_flat = spatial_features.view(batch_size, -1)
        spatial_dist = torch.cdist(spatial_features_flat, spatial_features_flat)
        
        if torch.isnan(spatial_dist).any() or torch.isinf(spatial_dist).any():
            spatial_dist = torch.zeros(batch_size, batch_size, device=device)
        
        spatial_dist_max = spatial_dist.max()
        if spatial_dist_max > 0:
            spatial_dist = spatial_dist / spatial_dist_max
        
        semantic_dist = torch.cdist(semantic_features, semantic_features)
        
        if torch.isnan(semantic_dist).any() or torch.isinf(semantic_dist).any():
            semantic_dist = torch.zeros(batch_size, batch_size, device=device)
        
        semantic_dist_max = semantic_dist.max()
        if semantic_dist_max > 0:
            semantic_dist = semantic_dist / semantic_dist_max
        
        combined_dist = (1 - self.feature_weight) * spatial_dist + self.feature_weight * semantic_dist
        
        edge_index_knn = []
        edge_index_radius = []
        
        for i in range(batch_size):
            k_neighbors = min(self.k + 1, batch_size)
            _, nn_idx = combined_dist[i].topk(k_neighbors, largest=False)
            
            if k_neighbors > 1:
                for j in nn_idx[1:]:
                    edge_index_knn.extend([[i, j.item()], [j.item(), i]])
                
            radius_neighbors = torch.where(combined_dist[i] < self.radius)[0]
            for j in radius_neighbors:
                if i != j:
                    edge_index_radius.extend([[i, j.item()], [j.item(), i]])
        
        edges = edge_index_knn + edge_index_radius
        if edges:
            edges = list(set(map(tuple, edges)))  
            edge_index = torch.tensor(edges, device=device).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        
        return edge_index


class ImprovedGCN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], output_dim=1024, dropout=0.2):
        super(ImprovedGCN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.layers.append(GCNConv(input_dim, hidden_dims[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        self.layers.append(GCNConv(hidden_dims[-1], output_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x, edge_index):
        identity = self.residual_conv(x)
        
        for i, (conv, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i != len(self.layers) - 1:  
                x = self.dropout(x)
        
        x = x + identity
        return F.relu(x)
