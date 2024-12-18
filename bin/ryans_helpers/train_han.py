#!/usr/bin/env python
import os
import json
import time
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from HAN import HAN
from data_processor import GenomicDataset

def save_checkpoint(state, filename):
    """Save checkpoint with a timestamp to avoid conflicts"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    """Load checkpoint if it exists"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        return torch.load(filename)
    return None

def update_metrics_log(metrics, filename):
    """Update metrics log file"""
    with open(filename, 'a') as f:
        json.dump(metrics, f)
        f.write('\n')

def mark_as_completed(output_dir):
    """Create a completion file to indicate training finished successfully"""
    completion_file = os.path.join(output_dir, 'TRAINING_COMPLETED')
    with open(completion_file, 'w') as f:
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Created completion marker: {completion_file}")

@click.command()
@click.option('--data-dir', required=True, help='Directory containing the processed data')
@click.option('--output-dir', required=True, help='Directory to save model checkpoints and metrics')
@click.option('--batch-size', default=32, help='Batch size for training')
@click.option('--num-epochs', default=100, help='Number of epochs to train')
@click.option('--learning-rate', default=0.001, help='Learning rate for optimizer')
@click.option('--num-motif-filters', default=128, help='Number of filters in motif detection module')
@click.option('--kmer-size', default=6, help='Size of k-mers for motif detection')
@click.option('--phrase-gru-size', default=64, help='Hidden size of phrase-level GRU')
@click.option('--sent-gru-size', default=64, help='Hidden size of sentence-level GRU')
@click.option('--num-targets', default=1, help='Number of target values to predict')
def train_han(data_dir, output_dir, batch_size, num_epochs, learning_rate,
              num_motif_filters, kmer_size, phrase_gru_size, sent_gru_size, num_targets):
    """Train the Hierarchical Attention Network for genomic sequence analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if training was already completed
    completion_file = os.path.join(output_dir, 'TRAINING_COMPLETED')
    if os.path.exists(completion_file):
        print(f"Training was already completed. Check {completion_file}")
        return
    
    # Initialize metrics tracking
    metrics_file = os.path.join(output_dir, 'metrics.jsonl')
    checkpoint_file = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    
    # Load datasets
    train_dataset = GenomicDataset(data_dir, split='train')
    val_dataset = GenomicDataset(data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, optimizer, and criterion
    model = HAN(
        num_motif_filters=num_motif_filters,
        kmer_size=kmer_size,
        phrase_gru_hidden_size=phrase_gru_size,
        sent_gru_hidden_size=sent_gru_size,
        num_targets=num_targets
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Load checkpoint if it exists
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
    
    # Training loop
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0
            train_start_time = time.time()
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences, targets = sequences.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Save intermediate checkpoint every N batches
                if (batch_idx + 1) % 100 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss / (batch_idx + 1),
                        'best_val_loss': best_val_loss
                    }
                    save_checkpoint(checkpoint, checkpoint_file)
            
            # Validation
            model.eval()
            val_loss = 0
            val_start_time = time.time()
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences, targets = sequences.to(device), targets.to(device)
                    outputs = model(sequences)
                    val_loss += criterion(outputs, targets).item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update metrics log
            metrics = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_time': time.time() - train_start_time,
                'val_time': time.time() - val_start_time,
                'timestamp': time.time()
            }
            update_metrics_log(metrics, metrics_file)
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }
                save_checkpoint(checkpoint, checkpoint_file)
                save_checkpoint(checkpoint, os.path.join(checkpoint_dir, f'best_model.pt'))
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {avg_val_loss:.6f}")
        
        # Mark training as completed
        mark_as_completed(output_dir)
    
    except (KeyboardInterrupt, SystemExit):
        print("Training interrupted. Saving checkpoint...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        save_checkpoint(checkpoint, checkpoint_file)
        raise

if __name__ == '__main__':
    train_han() 