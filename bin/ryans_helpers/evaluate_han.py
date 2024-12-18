#!/usr/bin/env python
import os
import json
import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from HAN import HAN
from data_processor import GenomicDataset

def load_checkpoint(filename):
    """Load checkpoint if it exists"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        return torch.load(filename)
    return None

def save_results(results, filename):
    """Save evaluation results to a file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def check_training_completed(model_dir):
    """Check if training has completed successfully"""
    completion_file = os.path.join(model_dir, 'TRAINING_COMPLETED')
    if not os.path.exists(completion_file):
        print(f"Training has not completed yet. Waiting for {completion_file}")
        return False
    with open(completion_file, 'r') as f:
        completion_time = f.read().strip()
    print(f"Found completed training: {completion_time}")
    return True

@click.command()
@click.option('--model-dir', required=True, help='Directory containing the model checkpoints')
@click.option('--data-dir', required=True, help='Directory containing the test data')
@click.option('--output-dir', required=True, help='Directory to save evaluation results')
@click.option('--batch-size', default=32, help='Batch size for evaluation')
@click.option('--num-motif-filters', default=128, help='Number of filters in motif detection module')
@click.option('--kmer-size', default=6, help='Size of k-mers for motif detection')
@click.option('--phrase-gru-size', default=64, help='Hidden size of phrase-level GRU')
@click.option('--sent-gru-size', default=64, help='Hidden size of sentence-level GRU')
@click.option('--num-targets', default=1, help='Number of target values to predict')
@click.option('--force', is_flag=True, help='Run evaluation even if training is not marked as completed')
def evaluate_han(model_dir, data_dir, output_dir, batch_size, num_motif_filters,
                kmer_size, phrase_gru_size, sent_gru_size, num_targets, force):
    """Evaluate the Hierarchical Attention Network on test data."""
    # Check if training has completed
    if not force and not check_training_completed(model_dir):
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the best model checkpoint
    checkpoint_file = os.path.join(model_dir, 'checkpoints/best_model.pt')
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint is None:
        print(f"No checkpoint found at {checkpoint_file}")
        return
    
    # Initialize model and load state
    model = HAN(
        num_motif_filters=num_motif_filters,
        kmer_size=kmer_size,
        phrase_gru_hidden_size=phrase_gru_size,
        sent_gru_hidden_size=sent_gru_size,
        num_targets=num_targets
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataset
    test_dataset = GenomicDataset(data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    criterion = nn.MSELoss(reduction='none')
    
    # Evaluation metrics
    all_losses = []
    all_predictions = []
    all_targets = []
    
    try:
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                
                # Calculate per-sample loss
                losses = criterion(outputs, targets)
                
                all_losses.extend(losses.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert lists to numpy arrays
        all_losses = np.array(all_losses)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        results = {
            'mean_loss': float(np.mean(all_losses)),
            'std_loss': float(np.std(all_losses)),
            'median_loss': float(np.median(all_losses)),
            'mean_absolute_error': float(np.mean(np.abs(all_predictions - all_targets))),
            'r2_score': float(1 - np.sum((all_targets - all_predictions) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2))
        }
        
        # Save results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        save_results(results, results_file)
        
        # Save detailed predictions
        detailed_results = {
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist(),
            'losses': all_losses.tolist()
        }
        detailed_results_file = os.path.join(output_dir, 'detailed_results.json')
        save_results(detailed_results, detailed_results_file)
        
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.6f}")
    
    except (KeyboardInterrupt, SystemExit):
        print("\nEvaluation interrupted. Saving partial results...")
        if len(all_losses) > 0:
            partial_results = {
                'mean_loss': float(np.mean(all_losses)),
                'num_samples_evaluated': len(all_losses),
                'total_samples': len(test_dataset),
                'completion_percentage': 100 * len(all_losses) / len(test_dataset)
            }
            partial_results_file = os.path.join(output_dir, 'partial_results.json')
            save_results(partial_results, partial_results_file)
        raise

if __name__ == '__main__':
    evaluate_han() 