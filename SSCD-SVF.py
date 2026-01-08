#!/usr/bin/env python3
"""
CDCD Dual-Modal DDP Training with Adaptive GPK Dimensions
Complete version with all training curves and ROC data saving
"""
import os
import re
import csv
import time
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from common modules
from modules import (
    normalize_chromosome_name,
    extract_chr_from_filename,
    extract_id_from_filename
)

# Import from loss function module
from loss_function import (
    AdaptiveCDCDDualModalModel,  # Adaptive model
    IntelligentGPKDataLoader,    # Intelligent GPK loader
    PriorModulatedContrastiveLoss,
    LogicConsistencyConstraint,
    MACLoss,
    LearnableLossWeights
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# Argument Parsing
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Adaptive GPK Dimension CDCD Dual-Modal DDP Training Script')
    
    # Data paths
    parser.add_argument('--bases_root', required=True, 
                       help='Root directory for bases images with three class folders')
    parser.add_argument('--cigar_root', required=True, 
                       help='Root directory for cigar images with three class folders')
    parser.add_argument('--class_dirs', nargs=3, required=True,
                       metavar=('CLS0', 'CLS1', 'CLS2'),
                       help='Three class folder names, order=label 0/1/2')
    
    # GPK data path
    parser.add_argument('--gpk_csv', required=True,
                       help='CSV file path for GPK statistics')
    
    # Chromosome splitting
    parser.add_argument('--train_chrs', nargs='+', 
                       default=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                       help='Training chromosome names')
    parser.add_argument('--test_chrs', nargs='+', 
                       default=['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y'],
                       help='Test chromosome names')
    
    # Model configuration
    parser.add_argument('--bases_model', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet_v2'],
                       help='Model used for bases modality')
    parser.add_argument('--cigar_model', default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet_v2'],
                       help='Model used for cigar modality')
    
    # CDCD configuration
    parser.add_argument('--use_cdcd', action='store_true', default=True,
                       help='Use CDCD module')
    parser.add_argument('--use_mac', action='store_true', default=True,
                       help='Use MAC loss')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                       help='Weight decay coefficient')
    
    # Early stopping parameters
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                       help='Minimum improvement threshold for early stopping')
    
    # Layer-wise fine-tuning
    parser.add_argument('--freeze_layers', type=int, default=5,
                       help='Number of bottom convolutional layers to freeze')
    parser.add_argument('--trainable_layers', type=int, default=3,
                       help='Number of high-level convolutional layers to train')
    
    # MAC loss parameters
    parser.add_argument('--init_alpha', type=float, default=0.1,
                       help='Initial weight for PMCL loss')
    parser.add_argument('--init_beta', type=float, default=0.1,
                       help='Initial weight for LCON loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature parameter for PMCL')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Kernel scaling factor for PMCL')
    
    # DDP parameters
    parser.add_argument('--world_size', type=int, default=1,
                       help='Total number of processes')
    parser.add_argument('--dist_url', default='env://',
                       help='Distributed training URL')
    parser.add_argument('--dist_backend', default='nccl',
                       help='Distributed training backend')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='DDP local rank')
    
    # Save path
    parser.add_argument('--save_path', required=True,
                       help='Model save path')
    
    return parser.parse_args()


# =========================================================
# DDP Initialization
# =========================================================
def init_distributed_mode(args):
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif hasattr(args, 'local_rank') and args.local_rank != -1:
        args.rank = args.local_rank
        args.world_size = 1
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return args
    
    args.distributed = True
    
    # Set CUDA device
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f'cuda:{args.local_rank}')
    
    # Initialize process group
    print(f'| distributed init (rank {args.rank}): {args.dist_url}')
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    
    dist.barrier()
    
    return args


def initialize_model(args, actual_gpk_dim):
    """Initialize model"""
    if args.rank == 0:
        print(f"[INFO] Initializing model on device {args.device}...")
        print(f"[INFO] GPK dimension: {actual_gpk_dim}")
    
    # Create model
    model = AdaptiveCDCDDualModalModel(
        bases_model_name=args.bases_model,
        cigar_model_name=args.cigar_model,
        num_classes=3,
        freeze_layers=args.freeze_layers,
        trainable_layers=args.trainable_layers,
        projection_dim=512,
        gpk_dim=actual_gpk_dim,
        use_cdcd=args.use_cdcd,
        use_mac=args.use_mac,
        init_alpha=args.init_alpha,
        init_beta=args.init_beta,
        temperature=args.temperature,
        gamma=args.gamma,
        rank=args.rank
    ).to(args.device)
    
    # Initialize with actual GPK dimension
    model.initialize_with_actual_gpk_dim(actual_gpk_dim)
    
    if args.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    return model


# =========================================================
# Random Seed Setting
# =========================================================
def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Data Augmentation
# =========================================================
def build_transforms():
    """Build data augmentation transforms"""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_tf, val_tf


# =========================================================
# Adaptive GPK Dataset
# =========================================================

class AdaptiveGPKDataset(Dataset):
    """
    Adaptive GPK Dataset
    Supports multiple ID formats and automatic GPK vector retrieval
    """
    def __init__(self, 
                 bases_root: str,
                 cigar_root: str,
                 class_dirs: List[str],
                 chr_list: List[str],
                 gpk_loader: IntelligentGPKDataLoader,
                 transform=None,
                 args=None):
        self.transform = transform
        self.args = args
        self.gpk_loader = gpk_loader
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        self.samples = []
        
        # Collect all files
        self._collect_samples(bases_root, cigar_root, class_dirs, chr_list)
        
        if args and args.rank == 0:
            print(f"[INFO] Dataset loading completed: {len(self.samples)} samples")
            if self.samples:
                print(f"[INFO] Sample examples:")
                for i in range(min(3, len(self.samples))):
                    bases_path, cigar_path, label, file_id, chr_name = self.samples[i]
                    print(f"  Sample {i+1}: ID={file_id}, Label={label}, Chromosome={chr_name}")
    
    def _collect_samples(self, bases_root, cigar_root, class_dirs, chr_list):
        """Collect aligned samples"""
        bases_samples = []
        cigar_samples = {}
        
        # Define ID extraction function
        def extract_file_id_optimized(filename):
            stem = Path(filename).stem
            pattern = r'([0-9XY]+)\.[0-9]+\.[A-Z]+\.[0-9]+'
            match = re.search(pattern, stem, re.IGNORECASE)
            if match:
                full_id = match.group(0)
                if full_id.startswith('chr_'):
                    full_id = full_id[4:]
                elif full_id.startswith('chr'):
                    full_id = full_id[3:]
                return full_id
            
            parts = stem.split('.')
            if len(parts) >= 4:
                id_parts = parts[:4]
                if id_parts[3].isdigit():
                    return '.'.join(id_parts)
            
            parts = stem.split('_')
            if len(parts) >= 4:
                return '_'.join(parts[:4])
            
            return stem.replace('.bases', '').replace('.cigar', '')
        
        # Collect bases images
        for cls_idx, cls_name in enumerate(class_dirs):
            cls_dir = Path(bases_root) / cls_name
            if cls_dir.exists():
                files = list(cls_dir.glob('*.png'))
                for file in files:
                    chr_name = extract_chr_from_filename(file.name)
                    
                    if chr_name and chr_name in chr_list:
                        file_id = extract_file_id_optimized(file.name)
                        if file_id:
                            bases_samples.append((
                                str(file),
                                cls_idx,
                                file_id,
                                cls_name,
                                chr_name
                            ))
        
        # Collect cigar images
        for cls_idx, cls_name in enumerate(class_dirs):
            cls_dir = Path(cigar_root) / cls_name
            if cls_dir.exists():
                files = list(cls_dir.glob('*.png'))
                for file in files:
                    file_id = extract_file_id_optimized(file.name)
                    if file_id:
                        cigar_samples[file_id] = str(file)
        
        # Align samples
        matched_count = 0
        for bases_path, label, file_id, cls_name, chr_name in bases_samples:
            if file_id in cigar_samples:
                self.samples.append((
                    bases_path,
                    cigar_samples[file_id],
                    label,
                    file_id,
                    chr_name
                ))
                matched_count += 1
        
        if self.args and self.args.rank == 0:
            print(f"[INFO] Sample matching: {matched_count}/{len(bases_samples)} bases samples found corresponding cigar samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bases_path, cigar_path, label, file_id, chr_name = self.samples[idx]
        
        # Load images
        bases_img = datasets.folder.default_loader(bases_path)
        cigar_img = datasets.folder.default_loader(cigar_path)
        
        if self.transform:
            bases_img = self.transform(bases_img)
            cigar_img = self.transform(cigar_img)
        
        # Get GPK vector
        gpk_vector = self.gpk_loader.get_gpk_vector(file_id)
        gpk_vector = torch.FloatTensor(gpk_vector)
        
        return (bases_img, cigar_img, gpk_vector), label


# =========================================================
# Function to Print Evaluation Results
# =========================================================

def print_evaluation_results(epoch, total_epochs, lr, 
                           train_loss, train_acc, train_pre, train_rec, train_f1,
                           val_loss, val_acc, val_pre, val_rec, val_f1,
                           y_true=None, y_pred=None, epoch_time=None):
    """Print formatted evaluation results"""
    
    print(f"\nEpoch {epoch:02d}/{total_epochs}, lr: {lr:.2e}")
    print("=" * 80)
    
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Metrics -> ACC: {train_acc:.4f} | PRE: {train_pre:.4f} | REC: {train_rec:.4f} | F1: {train_f1:.4f}")
    print("-" * 80)
    
    print(f"Val   Loss : {val_loss:.4f}")
    print(f"Val Metrics -> ACC: {val_acc:.4f} | PRE: {val_pre:.4f} | REC: {val_rec:.4f} | F1: {val_f1:.4f}")
    
    if y_true is not None and y_pred is not None:
        num_classes = len(np.unique(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        print(f"\nConfusion Matrix:")
        print(" " * 10 + "".join([f"Pred {i:<8}" for i in range(num_classes)]))
        
        for i in range(num_classes):
            row_str = f"True {i}:" + "".join([f"{cm[i, j]:<10}" for j in range(num_classes)])
            print(row_str)
        
        print(f"\nClass-wise Metrics:")
        print(f"{'Class':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 50)
        
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            pre = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
            
            print(f"{i:<10} {acc:<10.4f} {pre:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    
    if epoch_time is not None:
        print(f"\nEpoch Time: {epoch_time:.2f} sec")
    
    print("=" * 80)


# =========================================================
# Function to Save ROC Data
# =========================================================

def save_roc_data(args, y_true, y_pred, y_prob, file_path=None):
    """Save ROC data to CSV file"""
    if y_prob.size == 0:
        print("[WARNING] No probability data to save for ROC")
        return
    
    roc_data = []
    for i in range(len(y_true)):
        row = {
            'true_label': int(y_true[i]),
            'predicted_class': int(y_pred[i])
        }
        for class_idx in range(y_prob.shape[1]):
            row[f'prob_class_{class_idx}'] = float(y_prob[i, class_idx])
        roc_data.append(row)
    
    roc_df = pd.DataFrame(roc_data)
    
    if file_path is None:
        model_name = Path(args.save_path).stem
        roc_csv_path = f"{model_name}.roc.csv"
    else:
        roc_csv_path = file_path
    
    output_dir = os.path.dirname(roc_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    roc_df.to_csv(roc_csv_path, index=False)
    
    if args.rank == 0:
        print(f'ROC data saved to: {roc_csv_path}')
        print(f'ROC data shape: {roc_df.shape}')
    
    return roc_csv_path


# =========================================================
# Early Stopping Class
# =========================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_f1, model):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict() if hasattr(model, 'state_dict') else None
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict() if hasattr(model, 'state_dict') else None
            self.counter = 0

        return self.early_stop


# =========================================================
# Training Loop
# =========================================================

def train_epoch_with_adaptive_cdcd(model, loader, criterion, optimizer, scaler, device, args, epoch: int):
    """Train one epoch"""
    model.train()
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_pmcl_loss = 0.0
    total_lcon_loss = 0.0
    total_num = 0
    
    all_y, all_p = [], []
    
    for batch_idx, ((x1, x2, gpk), y) in enumerate(loader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        gpk = gpk.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # Key fix: Correctly access the original model
            # In DDP mode, model is DDP wrapper, need to access original model via model.module
            original_model = model.module if hasattr(model, 'module') else model
            
            # Forward pass
            outputs = original_model(x1, x2, gpk_vector=gpk, labels=y, return_features=True)
            
            # Compute loss
            if args.use_mac:
                step = epoch * len(loader) + batch_idx
                # Key fix: Call compute_mac_loss on original model
                loss_dict = original_model.compute_mac_loss(
                    outputs, y, gpk_vectors=gpk, step=step, return_components=True
                )
                loss = loss_dict['total']
            else:
                loss = criterion(outputs['primary_logits'], y)
                loss_dict = {'task': loss, 'total': loss}
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_task_loss += loss_dict['task'].item() * bs
        total_pmcl_loss += loss_dict.get('pmcl', torch.tensor(0.0)).item() * bs
        total_lcon_loss += loss_dict.get('lcon', torch.tensor(0.0)).item() * bs
        total_num += bs
        
        all_y.append(y.detach().cpu())
        all_p.append(outputs['primary_logits'].argmax(1).detach().cpu())
    
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    
    metrics = {
        'total_loss': total_loss / max(total_num, 1),
        'task_loss': total_task_loss / max(total_num, 1),
        'pmcl_loss': total_pmcl_loss / max(total_num, 1),
        'lcon_loss': total_lcon_loss / max(total_num, 1),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    return metrics


# =========================================================
# Evaluation Function
# =========================================================

@torch.no_grad()
@torch.no_grad()
def evaluate_adaptive_cdcd(model, loader, criterion, device, args):
    """Evaluation"""
    model.eval()
    
    all_y, all_p, all_probs, losses = [], [], [], []
    all_confidence_str, all_confidence_seq = [], []
    
    for (x1, x2, gpk), y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        gpk = gpk.to(device)
        y = y.to(device)
        
        # Key fix: Correctly access original model for evaluation
        original_model = model.module if hasattr(model, 'module') else model
        outputs = original_model(x1, x2, gpk_vector=gpk, return_features=True)
        
        logits = outputs['primary_logits']
        
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        
        all_y.append(y.cpu())
        all_p.append(logits.argmax(1).cpu())
        all_probs.append(probs.cpu())
        losses.append(loss.item())
        
        # Record confidence
        if 'confidence_str' in outputs:
            all_confidence_str.append(outputs['confidence_str'].cpu())
        if 'confidence_seq' in outputs:
            all_confidence_seq.append(outputs['confidence_seq'].cpu())
    
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    y_prob = torch.cat(all_probs).numpy() if all_probs else np.array([])
    avg_loss = np.mean(losses) if losses else 0.0
    
    # Confidence statistics
    confidence_stats = {}
    if all_confidence_str:
        confidence_str = torch.cat(all_confidence_str).numpy()
        confidence_stats['confidence_str_mean'] = np.mean(confidence_str)
        confidence_stats['confidence_str_std'] = np.std(confidence_str)
    
    if all_confidence_seq:
        confidence_seq = torch.cat(all_confidence_seq).numpy()
        confidence_stats['confidence_seq_mean'] = np.mean(confidence_seq)
        confidence_stats['confidence_seq_std'] = np.std(confidence_seq)
    
    return avg_loss, y_true, y_pred, y_prob, confidence_stats


# =========================================================
# Main Function - Complete Adaptive Version
# =========================================================

def main():
    """Main function - Adaptive GPK dimension version"""
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Initialize distributed training
    args = init_distributed_mode(args)
    
    if args.rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Adaptive GPK Dimension CDCD Dual-Modal Training")
        print(f"Using device: {args.device}, world_size={args.world_size}")
        print(f"Bases model: {args.bases_model}, Cigar model: {args.cigar_model}")
        print(f"Use CDCD: {args.use_cdcd}, Use MAC: {args.use_mac}")
        print(f"Training epochs: {args.epochs}, Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
        print(f"{'=' * 80}")
    
    # 3. Set random seed
    set_seed(args.seed + args.rank)
    
    # 4. Load GPK data
    if args.rank == 0:
        print(f"\n[INFO] Loading GPK data...")
    
    gpk_loader = IntelligentGPKDataLoader(
        csv_path=args.gpk_csv,
        normalize=True,
        fill_missing=True
    )
    gpk_loader.load_data()
    
    # Get actual GPK dimension
    actual_gpk_dim = gpk_loader.gpk_dim
    gpk_info = gpk_loader.get_info()
    
    if args.rank == 0:
        print(f"[INFO] GPK information:")
        print(f"  Number of samples: {gpk_info['num_samples']}")
        print(f"  Feature dimension: {gpk_info['gpk_dim']}")
        print(f"  Feature column examples: {gpk_info['columns'][:5]}...")
        print(f"  Data statistics: {gpk_info['statistics']}")
    
    # 5. Normalize chromosome names
    normalized_train_chrs = [normalize_chromosome_name(c) for c in args.train_chrs if normalize_chromosome_name(c)]
    normalized_test_chrs = [normalize_chromosome_name(c) for c in args.test_chrs if normalize_chromosome_name(c)]
    
    if args.rank == 0:
        print(f"\n[INFO] Normalized training chromosomes: {normalized_train_chrs}")
        print(f"[INFO] Normalized test chromosomes: {normalized_test_chrs}")
    
    # 6. Create datasets and loaders
    if args.rank == 0:
        print(f"\n[INFO] Creating datasets...")
    
    train_tf, val_tf = build_transforms()
    
    train_ds = AdaptiveGPKDataset(
        args.bases_root, args.cigar_root, args.class_dirs,
        normalized_train_chrs, gpk_loader, train_tf, args
    )
    
    val_ds = AdaptiveGPKDataset(
        args.bases_root, args.cigar_root, args.class_dirs,
        normalized_test_chrs, gpk_loader, val_tf, args
    )
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        if args.rank == 0:
            print("[ERROR] No aligned samples found!")
            print(f"Training set size: {len(train_ds)}")
            print(f"Validation set size: {len(val_ds)}")
        return
    
    # Distributed samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if args.distributed else None
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=train_sampler is None,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True if args.distributed else False
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    if args.rank == 0:
        print(f"[INFO] Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        print(f"[INFO] Training batches/epoch: {len(train_loader)}, Validation batches/epoch: {len(val_loader)}")
    
    # 7. Create adaptive model
    model = initialize_model(args, actual_gpk_dim)
    
    # 8. DDP wrapper
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # 9. Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # 10. Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.min_delta,
        verbose=(args.rank == 0)
    )
    
    # 11. Generate output file paths
    save_path = Path(args.save_path)
    model_name = save_path.stem
    save_dir = save_path.parent if save_path.parent != Path('.') else Path('.')
    
    if args.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    curve_csv_path = str(save_dir / f"{model_name}.curve.csv")
    roc_csv_path = str(save_dir / f"{model_name}.roc.csv")
    
    if args.rank == 0:
        print(f"\n[INFO] Model will be saved to: {args.save_path}")
        print(f"[INFO] Training curves will be saved to: {curve_csv_path}")
        print(f"[INFO] ROC data will be saved to: {roc_csv_path}")
        print(f"{'=' * 80}\n")
    
    # 12. Training loop
    curve_data = []
    best_f1 = 0.0
    best_model_state = None
    best_roc_data = None
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Training
        train_metrics = train_epoch_with_adaptive_cdcd(
            model, train_loader, criterion, optimizer, scaler, args.device, args, epoch
        )
        
        # Evaluation
        val_loss, y_true, y_pred, y_prob, confidence_stats = evaluate_adaptive_cdcd(
            model,  # Pass model directly, let evaluate_adaptive_cdcd handle internally
            val_loader, criterion, args.device, args
        )
        
        if args.rank == 0:
            epoch_time = time.time() - epoch_start_time
            
            val_acc = accuracy_score(y_true, y_pred)
            val_pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
            val_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record training curve data
            curve_data.append({
                'Epoch': epoch,
                'Learning_Rate': current_lr,
                'Train_Loss': train_metrics['total_loss'],
                'Train_ACC': train_metrics['accuracy'],
                'Train_PRE': train_metrics['precision'],
                'Train_REC': train_metrics['recall'],
                'Train_F1': train_metrics['f1'],
                'Val_Loss': val_loss,
                'Val_ACC': val_acc,
                'Val_PRE': val_pre,
                'Val_REC': val_rec,
                'Val_F1': val_f1,
                'Task_Loss': train_metrics['task_loss'],
                'PMCL_Loss': train_metrics['pmcl_loss'],
                'LCON_Loss': train_metrics['lcon_loss']
            })
            
            # Print evaluation results
            print_evaluation_results(
                epoch=epoch,
                total_epochs=args.epochs,
                lr=current_lr,
                train_loss=train_metrics['total_loss'],
                train_acc=train_metrics['accuracy'],
                train_pre=train_metrics['precision'],
                train_rec=train_metrics['recall'],
                train_f1=train_metrics['f1'],
                val_loss=val_loss,
                val_acc=val_acc,
                val_pre=val_pre,
                val_rec=val_rec,
                val_f1=val_f1,
                y_true=y_true,
                y_pred=y_pred,
                epoch_time=epoch_time
            )
            
            # Print confidence information
            if args.use_cdcd and confidence_stats:
                print(f"[INFO] Confidence statistics - "
                      f"Structure modality: {confidence_stats.get('confidence_str_mean', 0):.3f}±"
                      f"{confidence_stats.get('confidence_str_std', 0):.3f}, "
                      f"Sequence modality: {confidence_stats.get('confidence_seq_mean', 0):.3f}±"
                      f"{confidence_stats.get('confidence_seq_std', 0):.3f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.module.state_dict() if args.distributed else model.state_dict()
                best_roc_data = (y_true, y_pred, y_prob)
                
                torch.save(best_model_state, args.save_path)
                print(f"  [INFO] Saving best model! Validation F1: {val_f1:.4f}")
            
            # Early stopping check
            if early_stopping(val_f1, model.module if args.distributed else model):
                print(f"\n{'=' * 80}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best validation F1: {best_f1:.4f}")
                
                if best_model_state is not None:
                    if args.distributed:
                        model.module.load_state_dict(best_model_state)
                    else:
                        model.load_state_dict(best_model_state)
                    torch.save(best_model_state, args.save_path)
                break
        
        scheduler.step()
    
    # 13. Save training results
    if args.rank == 0:
        # Save training curve data
        curve_df = pd.DataFrame(curve_data)
        curve_df.to_csv(curve_csv_path, index=False)
        print(f'\nTraining curve data saved to: {curve_csv_path}')
        print(f'Training curve data shape: {curve_df.shape}')
        
        # Save ROC data
        if best_roc_data is not None:
            y_true, y_pred, y_prob = best_roc_data
            save_roc_data(args, y_true, y_pred, y_prob, roc_csv_path)
        else:
            print("[WARNING] No ROC data to save")
        
        print(f"\n{'=' * 80}")
        print(f"Training completed!")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Model saved to: {args.save_path}")
        print(f"Training curves saved to: {curve_csv_path}")
        print(f"ROC data saved to: {roc_csv_path}")
        print('=' * 80)
    
    # 14. Clean up distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
