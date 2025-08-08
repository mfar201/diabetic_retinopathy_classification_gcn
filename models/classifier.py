import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import FeatureExtractor
from models.graph import GraphConstruction, ImprovedGCN
from models.fusion import CrossAttentionFusion, AdaptiveFusion
from config import FusionStrategy

class ImprovedMedicalImageClassifier(nn.Module):
    def __init__(self, feature_extractor, model_configs, num_classes=5,
                 fusion_strategy=FusionStrategy.ATTENTION, uncertainty=False):
        super(ImprovedMedicalImageClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_models = len(model_configs)
        self.fusion_strategy = fusion_strategy
        self.uncertainty = uncertainty
        
        self.base_dim = model_configs[0]['output_dim']
        
        self.graph_constructor = GraphConstruction(k=4, radius=0.1, feature_weight=0.5)
        
        self.gcn = ImprovedGCN(
            input_dim=self.base_dim,
            hidden_dims=[512, 256],
            output_dim=self.base_dim
        )
        
        self.quality_branch = nn.Sequential(
            nn.Linear(self.base_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        if self.num_models > 1:
            if self.fusion_strategy == FusionStrategy.ATTENTION:
                self.fusion = CrossAttentionFusion(
                    model_configs[0]['output_dim'],
                    model_configs[1]['output_dim']
                )
            elif self.fusion_strategy == FusionStrategy.ADAPTIVE:
                self.fusion = AdaptiveFusion(model_configs[0]['output_dim'], num_classes)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_dim, num_classes)
        )
        
        self.attention_weights = None
    
    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        
        features_list = self.feature_extractor(x)
        
        if self.num_models == 1:
            features = features_list[0]
        else:
            if self.fusion_strategy == FusionStrategy.ATTENTION:
                features = self.fusion(features_list[0], features_list[1])
            elif self.fusion_strategy == FusionStrategy.ADAPTIVE:
                features = self.fusion(features_list[0], features_list[1])
        
        edge_index = self.graph_constructor.create_edge_index(x, features)
        
        gcn_features = self.gcn(features, edge_index)
        
        quality_score = self.quality_branch(gcn_features)
        
        if self.training or not self.uncertainty:
            logits = self.classifier(gcn_features)
            weighted_logits = logits * quality_score
            
            if return_attention:
                return weighted_logits, None, quality_score, self.attention_weights
            return weighted_logits
        else:
            predictions = []
            for _ in range(10):
                pred = self.classifier(gcn_features)
                pred = pred * quality_score
                predictions.append(F.softmax(pred, dim=1))
            
            mean_pred = torch.stack(predictions).mean(0)
            uncertainty = torch.stack(predictions).std(0)
            
            if return_attention:
                return mean_pred, uncertainty, quality_score, self.attention_weights
            return mean_pred, uncertainty
