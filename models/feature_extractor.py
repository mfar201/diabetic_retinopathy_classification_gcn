import torch
import torch.nn as nn
import timm

class FeatureExtractor(nn.Module):
    def __init__(self, model_configs):
        super(FeatureExtractor, self).__init__()
        self.models = nn.ModuleList()
        self.output_dims = []
        self.types = []
        
        for config in model_configs:
            model = timm.create_model(config['name'], pretrained=True)
            # Reset classifier to get feature outputs
            if hasattr(model, 'reset_classifier'):
                model.reset_classifier(0)
            elif hasattr(model, 'head'):
                model.head = nn.Identity()
            self.models.append(model)
            self.output_dims.append(config['output_dim'])
            self.types.append(config['type'])

    def forward(self, x):
        features = []
        for model in self.models:
            feat = model(x)
            if len(feat.shape) > 2:
                feat = feat.mean([2, 3])  
            features.append(feat)
        return features
