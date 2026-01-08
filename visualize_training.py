#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_training_curves(csv_path, output_dir):
    """Plot training curves from CSV log"""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(df['Epoch'], df['Train_Loss'], label='Train')
    axes[0, 0].plot(df['Epoch'], df['Val_Loss'], label='Validation')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1-score
    axes[0, 1].plot(df['Epoch'], df['Train_F1'], label='Train')
    axes[0, 1].plot(df['Epoch'], df['Val_F1'], label='Validation')
    axes[0, 1].set_title('F1-Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy
    axes[0, 2].plot(df['Epoch'], df['Train_ACC'], label='Train')
    axes[0, 2].plot(df['Epoch'], df['Val_ACC'], label='Validation')
    axes[0, 2].set_title('Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Learning rate
    axes[1, 0].plot(df['Epoch'], df['Learning_Rate'])
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # Precision
    axes[1, 1].plot(df['Epoch'], df['Train_PRE'], label='Train')
    axes[1, 1].plot(df['Epoch'], df['Val_PRE'], label='Validation')
    axes[1, 1].set_title('Precision')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Recall
    axes[1, 2].plot(df['Epoch'], df['Train_REC'], label='Train')
    axes[1, 2].plot(df['Epoch'], df['Val_REC'], label='Validation')
    axes[1, 2].set_title('Recall')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Recall')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training curves')
    parser.add_argument('--csv_path', required=True, help='Path to training CSV file')
    parser.add_argument('--output_dir', default='.', help='Output directory for plots')
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_curves(args.csv_path, args.output_dir)
