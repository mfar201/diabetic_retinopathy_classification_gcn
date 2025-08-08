import torch
from enum import Enum

class FusionStrategy(Enum):
    ATTENTION = "attention"
    CONCATENATION = "concatenation"
    ADAPTIVE = "adaptive"

def get_default_config():
    config = {
        'models': [
            {
                'name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
                'output_dim': 768,
                'type': 'transformer'
            }
        ],
        'num_classes': 5,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 5e-5,
        'optimizer': 'ADAMW',
        'use_scheduler': True,
        'use_early_stopping': True,
        'patience': 15,
        'scheduler_patience': 7,
        'fusion_strategy': FusionStrategy.ATTENTION,
        'use_balancing_strategy': True,
        'balance_strategy': 'class_weights',
        'uncertainty': True,
        'graph_k': 4,
        'graph_radius': 0.1,
        'feature_weight': 0.5
    }
    
    return config

def get_data_paths():
    paths = {
        'train_dir': '',
        'val_dir': '',
        'test_dir': '',
        
        'finetune_dataset_train_dir': '/mnt/ocpc_ssd/projects/datasets/messidor_2/oversampled_split/train',
        'finetune_dataset_val_dir': '/mnt/ocpc_ssd/projects/datasets/messidor_2/oversampled_split/val',
        'finetune_dataset_test_dir': '/mnt/ocpc_ssd/projects/datasets/messidor_2/oversampled_split/test',
    }
    
    return paths

def get_class_names():
    return ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
