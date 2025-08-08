import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import sys
from datetime import datetime
from config import get_class_names
from models.feature_extractor import FeatureExtractor
from models.classifier import ImprovedMedicalImageClassifier
from utils.training import evaluate
from data.dataset import get_transforms
from torchvision.datasets import ImageFolder

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

def load_pretrained_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    feature_extractor = FeatureExtractor(config['models']).to(device)
    model = ImprovedMedicalImageClassifier(
        feature_extractor,
        config['models'],
        num_classes=config['num_classes'],
        fusion_strategy=config['fusion_strategy'],
        uncertainty=config['uncertainty']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded pre-trained model from {model_path}")
    # print(f"Original test metrics: {checkpoint['test_metrics']}")
    return model, config, checkpoint

def evaluate_model_on_dataset(model_path, test_dir, batch_size=32, num_workers=4, save_dir='saved_metrics'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    eval_log_path = os.path.join(save_dir, f'eval_{model_name}_{timestamp}.txt')
    metrics_json_path = os.path.join(save_dir, f'eval_{model_name}_{timestamp}.json')
    
    output_capture = OutputCapture(eval_log_path)
    original_stdout = sys.stdout
    sys.stdout = output_capture
    
    try:
        print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {model_path}")
        print(f"Test directory: {test_dir}")
        print("="*50)
        
        model, config, checkpoint = load_pretrained_model(model_path, device)
        
        class_names = get_class_names()
        
        _, eval_transform = get_transforms()
        
        test_dataset = ImageFolder(test_dir, transform=eval_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        criterion = nn.CrossEntropyLoss()
        
        test_metrics = evaluate(
            model, 
            test_loader, 
            criterion, 
            device, 
            class_names=class_names,
            print_report=True,
            print_confusion_matrix=True,
            plot_pr_curve=True
        )
        
        print("\nFinal Test Metrics:")
        print("="*30)
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        'model_path': model_path,
        'test_directory': test_dir,
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_config': make_json_serializable(config),
        'test_metrics': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                        for k, v in test_metrics.items()}
    }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Evaluation results saved to:")
    print(f"  - Log file: {eval_log_path}")
    print(f"  - Metrics JSON: {metrics_json_path}")
    
    return test_metrics

if __name__ == "__main__":
    model_path = os.path.join('saved_models', 'messidor_vit_base_patch16_224.augreg2_in21k_ft_in1k.pth')
    test_dir = '/mnt/ocpc_ssd/projects/datasets/messidor_2/oversampled_split/test/'  
    
    evaluate_model_on_dataset(model_path, test_dir)
