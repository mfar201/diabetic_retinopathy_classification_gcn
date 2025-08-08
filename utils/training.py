import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.metrics import calculate_metrics, plot_confusion_matrix, classification_report, plot_precision_recall_curve

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(train_loader, desc='Training', disable=True)
    for batch in progress_bar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs, uncertainty, quality_scores, attention_weights = model(images, return_attention=True)
        
        class_loss = criterion(outputs, labels)
        
        quality_loss = F.binary_cross_entropy(quality_scores, torch.ones_like(quality_scores))

        loss = class_loss + 0.1 * quality_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(train_loader)

    return metrics


def evaluate(model, val_loader, criterion, device, class_names=None, print_report=False, 
             print_confusion_matrix=False, plot_pr_curve=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Evaluating', disable=True)
        for batch in progress_bar:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs, uncertainty, quality_scores, attention_weights = model(images, return_attention=True)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    if print_confusion_matrix:
        plot_confusion_matrix(all_labels, all_preds, class_names)

    if print_report:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names if class_names else None))

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(val_loader)
    
    if plot_pr_curve:
        plot_precision_recall_curve(all_labels, all_probs, class_names)

    return metrics
