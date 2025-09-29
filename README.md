# Diabetic Retinopathy Classification with Graph Convolutional Networks (GCN)

This repository implements diabetic retinopathy (DR) classification using PyTorch and PyTorch Geometric (GCN). It provides scripts to train from scratch, fine-tune on another dataset, and evaluate a saved checkpoint. Reproducibility is facilitated through a centralized configuration, deterministic seeding, and saved training artifacts.

- Language: Python
- DL frameworks: PyTorch, TorchVision
- GNN: PyTorch Geometric
- Visualization/analysis: NumPy, Pandas, Matplotlib/Seaborn, scikit-learn

## Repository structure

- [config.py](https://github.com/mfar201/diabetic_retinopathy_classification_gcn/blob/51cb23a105f47f9d48cb1b595a0c274d65561911/config.py) — Central configuration for models, training, balancing, and data paths.
- [train.py](https://github.com/mfar201/diabetic_retinopathy_classification_gcn/blob/51cb23a105f47f9d48cb1b595a0c274d65561911/train.py) — Main training script; also evaluates on the test set and saves the best model and metrics.
- finetune.py — Fine-tune/continue training from a checkpoint or on another dataset.
- eval.py — Evaluate a trained checkpoint and export metrics/artifacts.
- [requirements.txt](https://github.com/mfar201/diabetic_retinopathy_classification_gcn/blob/51cb23a105f47f9d48cb1b595a0c274d65561911/requirements.txt) — Pinned dependencies (CUDA 11.8 builds).
- models/ — Model components (feature extractors, classifiers).
- utils/ — Training loop utilities, metrics, early stopping, plotting.
- data/
  - data/dataset.py — Dataset, transforms, loaders, and class-imbalance handling.

Saved artifacts:
- Models saved to: models/<model_name>.pth
- Logs/metrics saved to: saved_metrics/train_<model>_<timestamp>.txt and .json

## Installation

Python 3.10+ is recommended.

- Create a virtual environment
  - Conda:
    - conda create -n dr-gcn python=3.10 -y
    - conda activate dr-gcn
  - venv:
    - python -m venv .venv
    - source .venv/bin/activate  # Windows: .venv\Scripts\activate

- Install dependencies
  - GPU with CUDA 11.8 (recommended):
    - pip install -r requirements.txt
  - CPU-only or different CUDA:
    - Install PyTorch per your environment from [PyTorch website](https://pytorch.org/).
    - Install PyTorch Geometric per your Torch/CUDA combo from [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
    - Then install the remaining packages from requirements.txt (you may need to remove or adapt the torch/torchvision/torchaudio/torch-geometric lines).

Verification:
- python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

## Dataset preparation

This project assumes an ImageFolder-style layout (used in data/dataset.py):

- Example folder-per-class structure:
  - data/
    - train/
      - 0/
      - 1/
      - 2/
      - 3/
      - 4/
    - val/
      - 0/
      - 1/
      - 2/
      - 3/
      - 4/
    - test/
      - 0/
      - 1/
      - 2/
      - 3/
      - 4/

Notes:
- Labels should be consistent with five DR classes: 0–4. Class names returned by config.get_class_names(): ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"].

## Configuration

All core settings live in [config.py](https://github.com/mfar201/diabetic_retinopathy_classification_gcn/blob/51cb23a105f47f9d48cb1b595a0c274d65561911/config.py).

Set dataset paths in get_data_paths():
- train_dir, val_dir, test_dir — set these to your local dataset splits.
- finetune_dataset_* — optional alternate dataset paths for fine-tuning.

Example (edit config.py):
- def get_data_paths():
    return {
      'train_dir': 'data/train',
      'val_dir': 'data/val',
      'test_dir': 'data/test',
      'finetune_dataset_train_dir': '',
      'finetune_dataset_val_dir': '',
      'finetune_dataset_test_dir': '',
    }

## How to run

- Train from scratch:
  - python train.py
  - Behavior:
    - Seeds are fixed (42) for torch, numpy, and CUDA.
    - Data loaders come from data/dataset.py with transforms 
    - Best model (by validation F1) is retained and saved to models/.
    - Metrics history is plotted, and evaluation on the test set is performed automatically.
    - Artifacts:
      - models/<model_name>.pth
      - saved_metrics/train_<model>_<timestamp>.txt
      - saved_metrics/train_<model>_<timestamp>.json

- Fine-tune:
  - python finetune.py
  - Set finetune_dataset_* paths in config.get_data_paths() and/or point to a checkpoint as required by your workflow.

- Evaluate a checkpoint:
  - python eval.py
  - Make sure config paths and any checkpoint settings required by eval.py are correctly set.

## Reproducing the main result

1) Environment
- Use the provided [requirements.txt](https://github.com/mfar201/diabetic_retinopathy_classification_gcn/blob/51cb23a105f47f9d48cb1b595a0c274d65561911/requirements.txt) and a fresh virtual environment.

2) Dataset
- Prepare ImageFolder splits under data/train, data/val, data/test with classes 0–4
- Keep the same train/val/test splits across runs.

3) Configuration
- Edit [config.py](https://github.com/mfar201/diabetic_retinopathy_classification_gcn/blob/51cb23a105f47f9d48cb1b595a0c274d65561911/config.py):
  - Fill train_dir/val_dir/test_dir in get_data_paths().

4) Train
- python train.py
- The script saves logs and metrics to saved_metrics/ and the best checkpoint to models/.

5) Evaluate
- python eval.py 

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Public DR datasets (APTOS 2019, Messidor-2, and EyePACS)
