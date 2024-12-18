#!/usr/bin/env python
"""
Performance Visualization Script for HAN Model

This script generates visualizations of training and validation metrics:
1. Loss curves (training and validation)
2. Learning rate over time
3. Training time per epoch
4. Validation metrics over time
"""

import os
import json
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(metrics_file):
    """Load metrics from JSONL file"""
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line.strip()))
    return metrics

def plot_loss_curves(metrics, output_dir):
    """Plot training and validation loss curves"""
    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()

def plot_training_time(metrics, output_dir):
    """Plot training time per epoch"""
    epochs = [m['epoch'] for m in metrics]
    train_time = [m['train_time'] for m in metrics]
    val_time = [m['val_time'] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_time, label='Training Time', marker='o')
    plt.plot(epochs, val_time, label='Validation Time', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training and Validation Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_time.png'))
    plt.close()

def plot_loss_distribution(metrics, output_dir):
    """Plot distribution of losses"""
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train_loss, label='Training Loss')
    sns.kdeplot(val_loss, label='Validation Loss')
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.title('Distribution of Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_distribution.png'))
    plt.close()

def plot_convergence_analysis(metrics, output_dir):
    """Plot convergence analysis"""
    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    
    # Calculate rolling means
    window = 5
    train_rolling = np.convolve(train_loss, np.ones(window)/window, mode='valid')
    val_rolling = np.convolve(val_loss, np.ones(window)/window, mode='valid')
    epochs_rolling = epochs[window-1:]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_rolling, train_rolling, label='Training Loss (Rolling Mean)')
    plt.plot(epochs_rolling, val_rolling, label='Validation Loss (Rolling Mean)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Rolling Mean)')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'))
    plt.close()

@click.command()
@click.option('--metrics-file', required=True, help='Path to metrics JSONL file')
@click.option('--output-dir', required=True, help='Directory to save plots')
def main(metrics_file, output_dir):
    """Generate performance visualization plots for the HAN model."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics(metrics_file)
    
    # Set style
    plt.style.use('seaborn')
    
    # Generate plots
    plot_loss_curves(metrics, output_dir)
    plot_training_time(metrics, output_dir)
    plot_loss_distribution(metrics, output_dir)
    plot_convergence_analysis(metrics, output_dir)

if __name__ == '__main__':
    main() 