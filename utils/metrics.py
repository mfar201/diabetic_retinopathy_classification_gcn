import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, cohen_kappa_score,
    precision_recall_curve, average_precision_score, confusion_matrix,
    classification_report
)
import os

os.makedirs('saved_metrics', exist_ok=True)

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'auc-roc': roc_auc_score(y_true, y_prob, multi_class='ovr'),
        'auc-pr': average_precision_score(y_true, y_prob, average='macro'),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }

def plot_metrics(metrics_history):
    plt.figure(figsize=(15, 12))

    for i, metric in enumerate(metrics_history['train'].keys(), 1):
        plt.subplot(3, 2, i)
        plt.plot(metrics_history['train'][metric], label=f'Train {metric}')
        plt.plot(metrics_history['val'][metric], label=f'Val {metric}')
        plt.title(f'{metric.capitalize()} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.savefig('saved_metrics/metrics_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, class_names=None):
    n_classes = y_prob.shape[1]
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        avg_precision = average_precision_score(y_true == i, y_prob[:, i])
        
        label = f"{class_names[i] if class_names else f'Class {i}'} (AP = {avg_precision:.2f})"
        plt.plot(recall, precision, lw=4, label=label)

    plt.xlabel('Recall', fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.title('Precision-Recall Curve for Each Class', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc='best', fontsize=22)
    plt.grid(True)
    plt.savefig('saved_metrics/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    cm_normalized = cm.astype('float') / np.where(cm.sum(axis=1)[:, np.newaxis] != 0,
                                                 cm.sum(axis=1)[:, np.newaxis], 1)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=class_names if class_names else "auto",
                     yticklabels=class_names if class_names else "auto",
                     annot_kws={"size": 22}, cbar=True)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)

    plt.title('Normalized Confusion Matrix', fontsize=24)
    plt.ylabel('True Label', fontsize=24)
    plt.xlabel('Predicted Label', fontsize=24)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=22, rotation=45, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=22, rotation=0)

    plt.tight_layout()
    plt.savefig('saved_metrics/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names if class_names else "auto",
                     yticklabels=class_names if class_names else "auto",
                     annot_kws={"size": 22}, cbar=True)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)

    plt.title('Confusion Matrix (Raw Counts)', fontsize=24)
    plt.ylabel('True Label', fontsize=24)
    plt.xlabel('Predicted Label', fontsize=24)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=22, rotation=45, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=22, rotation=0)

    plt.tight_layout()

    plt.savefig('saved_metrics/confusion_matrix_raw.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_training_status(epoch, scheduler, early_stopping):
    print("\nTraining Status:")
    print(f"Early Stopping Counter: {early_stopping.counter}/{early_stopping.patience}")

    if early_stopping.best_loss is not None:
        print(f"Best Loss: {early_stopping.best_loss:.4f}")
    else:
        print("Best Loss: Not yet available")

    if scheduler:
        lr = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else [group['lr'] for group in
                                                                                scheduler.optimizer.param_groups]
        print(f"Learning Rate: {lr[0]:.6f}")
