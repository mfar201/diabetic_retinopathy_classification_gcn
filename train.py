import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import json
import sys
from datetime import datetime

from config import get_default_config, get_data_paths, get_class_names
from models.feature_extractor import FeatureExtractor
from models.classifier import ImprovedMedicalImageClassifier
from utils.early_stopping import EarlyStopping
from utils.training import train_epoch, evaluate
from utils.metrics import plot_metrics, print_training_status
from data.dataset import create_datasets, create_dataloaders


class OutputCapture:
    def __init__(self, file_path):
        self.file_path = file_path
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.log = open(file_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main(save_dir='saved_metrics'):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    config = get_default_config()
    paths = get_data_paths()
    class_names = get_class_names()
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['models'][0]['name'].replace('.', '_')
    
    train_log_path = os.path.join(save_dir, f'train_{model_name}_{timestamp}.txt')
    metrics_json_path = os.path.join(save_dir, f'train_{model_name}_{timestamp}.json')
    
    output_capture = OutputCapture(train_log_path)
    original_stdout = sys.stdout
    sys.stdout = output_capture
    
    try:
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {model_name}")
        print(f"Train directory: {paths['train_dir']}")
        print(f"Val directory: {paths['val_dir']}")
        print(f"Test directory: {paths['test_dir']}")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        train_dataset, val_dataset, test_dataset = create_datasets(
            paths['train_dir'],
            paths['val_dir'],
            paths['test_dir']
        )
        
        config.update({
            'num_workers': 4,
            'device': device
        })
        
        train_loader, val_loader, test_loader, criterion = create_dataloaders(
            train_dataset, val_dataset, test_dataset, config
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Number of classes: {config['num_classes']}")
        print(f"Class names: {class_names}")
        print("-" * 40)
        
        feature_extractor = FeatureExtractor(config['models']).to(device)
        model = ImprovedMedicalImageClassifier(
            feature_extractor,
            config['models'],
            num_classes=config['num_classes'],
            fusion_strategy=config['fusion_strategy'],
            uncertainty=config['uncertainty']
        ).to(device)
        
        print(f"Model architecture: {config['models']}")
        print(f"Fusion strategy: {config['fusion_strategy']}")
        print(f"Uncertainty estimation: {config['uncertainty']}")
        print("-" * 40)
        
        if config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=0.9, weight_decay=1e-4)
        elif config['optimizer'] == 'ADAMW':
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
        print(f"Optimizer: {config['optimizer']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Number of epochs: {config['num_epochs']}")
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=config['scheduler_patience'],
            verbose=True,
            min_lr=1e-7
        ) if config['use_scheduler'] else None
        
        early_stopping = EarlyStopping(
            patience=config['patience']
        ) if config['use_early_stopping'] else None
        
        print(f"Use scheduler: {config['use_scheduler']}")
        if config['use_scheduler']:
            print(f"Scheduler patience: {config['scheduler_patience']}")
        print(f"Use early stopping: {config['use_early_stopping']}")
        if config['use_early_stopping']:
            print(f"Early stopping patience: {config['patience']}")
        
        metrics_history = {
            'train': {'loss': [], 'accuracy': [], 'f1': [], 'auc-roc': [], 'auc-pr': [], 'kappa': []},
            'val': {'loss': [], 'accuracy': [], 'f1': [], 'auc-roc': [], 'auc-pr': [], 'kappa': []}
        }
        
        best_f1 = 0
        best_model_state = None
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print("-" * 30)
            
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            
            val_metrics = evaluate(model, val_loader, criterion, device, class_names=class_names)
            
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")
                scheduler.step(val_metrics['loss'])
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"Learning rate decreased to {new_lr}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_model_state = model.state_dict()
                print(f"New best F1-Score: {best_f1:.4f}")
            
            print_training_status(epoch, scheduler, early_stopping)
            
            for split, metrics in zip(['train', 'val'], [train_metrics, val_metrics]):
                for metric, value in metrics.items():
                    metrics_history[split][metric].append(value)
            
            print("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                print(f"  {metric}: {value:.4f}")

            print("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            if early_stopping is not None:
                early_stopping(val_metrics['loss'])
                if early_stopping.early_stop:
                    print("\nEarly stopping triggered")
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with validation F1-Score: {best_f1:.4f}")

        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)

        plot_metrics(metrics_history)

        print("\nEvaluating on test set...")
        print("-" * 30)
        test_metrics = evaluate(
            model, test_loader, criterion, device,
            class_names=class_names, 
            print_report=True, 
            print_confusion_matrix=True, 
            plot_pr_curve=True
        )
        
        print("\nFinal Test Metrics:")
        print("="*25)
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        os.makedirs('models', exist_ok=True)
        model_save_path = os.path.join('models', f"{model_name}.pth")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'metrics_history': metrics_history,
            'test_metrics': test_metrics
        }, model_save_path)
        
        print(f"\nModel saved to {model_save_path}")
        print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = original_stdout
        output_capture.close()
    
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    metrics_to_save = {
        'training_timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'directories': {
            'train': paths['train_dir'],
            'val': paths['val_dir'],
            'test': paths['test_dir']
        },
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'config': make_json_serializable(config),
        'class_names': class_names,
        'device': str(device),
        'best_validation_f1': float(best_f1),
        'metrics_history': {
            split: {metric: [float(v) for v in values] 
                   for metric, values in metrics.items()}
            for split, metrics in metrics_history.items()
        },
        'final_test_metrics': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                              for k, v in test_metrics.items()},
        'model_save_path': model_save_path
    }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"\nTraining results saved to:")
    print(f"  - Log file: {train_log_path}")
    print(f"  - Metrics JSON: {metrics_json_path}")
    
    return model, test_metrics


if __name__ == '__main__':
    main()
