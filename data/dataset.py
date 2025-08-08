import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn

def get_transforms():
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform


def create_datasets(train_dir, val_dir, test_dir):
    train_transform, eval_transform = get_transforms()
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=eval_transform)
    test_dataset = ImageFolder(test_dir, transform=eval_transform)
    
    return train_dataset, val_dataset, test_dataset


def create_balanced_training(train_dataset, config):
    train_labels = np.array([sample[1] for sample in train_dataset.samples])

    class_counts = np.bincount(train_labels)
    unique_classes = np.unique(train_labels)

    print("Class distribution:")
    for cls in unique_classes:
        print(f"Class {cls}: {class_counts[cls]} samples")

    if config['balance_strategy'] == 'class_weights':
        class_weights = compute_class_weight('balanced',
                                             classes=unique_classes,
                                             y=train_labels)
        print("\nClass weights:", class_weights)

        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(config['device'])
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )

    elif config['balance_strategy'] == 'weighted_sampler':
        weights = 1.0 / class_counts
        sample_weights = weights[train_labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=config['num_workers']
        )

        criterion = nn.CrossEntropyLoss()

    else:
        raise ValueError("balance_strategy must be either 'class_weights' or 'weighted_sampler'")

    return criterion, train_loader


def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    if config['use_balancing_strategy']:
        criterion, train_loader = create_balanced_training(train_dataset, config)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        criterion = nn.CrossEntropyLoss()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, criterion
