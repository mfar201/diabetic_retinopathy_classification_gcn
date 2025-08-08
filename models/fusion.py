import torch
import torch.nn as nn
import numpy as np

class AdaptiveFusion(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AdaptiveFusion, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, cnn_out, gcn_out):
        return (1 - self.adaptive_weight) * cnn_out + self.adaptive_weight * gcn_out


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=256):
        super(CrossAttentionFusion, self).__init__()
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        
        self.query1 = nn.Linear(hidden_dim, hidden_dim)
        self.key2 = nn.Linear(hidden_dim, hidden_dim)
        self.value2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim * 2, dim1)

    def forward(self, x1, x2):
        x1_proj = self.proj1(x1)
        x2_proj = self.proj2(x2)
        
        q1 = self.query1(x1_proj)
        k2 = self.key2(x2_proj)
        v2 = self.value2(x2_proj)

        attention_scores = torch.matmul(q1, k2.transpose(-2, -1)) / np.sqrt(k2.size(-1))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, v2)
        attended_features = self.norm(attended_features)

        combined = torch.cat([x1_proj, attended_features], dim=-1)
        return self.final_proj(combined)
