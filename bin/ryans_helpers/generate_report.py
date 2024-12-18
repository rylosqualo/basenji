#!/usr/bin/env python
"""
Report Generation Script for HAN Model Results

This script generates a comprehensive HTML report containing:
1. Model architecture summary
2. Training performance plots
3. Evaluation metrics
4. Prediction analysis
"""

import os
import json
import click
import base64
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_results(results_file):
    """Load evaluation results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def encode_image(image_path):
    """Encode image as base64 for HTML embedding"""
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def generate_prediction_plots(results, output_dir):
    """Generate additional plots for prediction analysis"""
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # Scatter plot of predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    scatter_path = os.path.join(output_dir, 'predictions_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    
    # Error distribution
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    error_dist_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(error_dist_path)
    plt.close()
    
    return scatter_path, error_dist_path

def create_metrics_table(metrics):
    """Create HTML table for metrics"""
    rows = []
    for metric, value in metrics.items():
        rows.append(f"<tr><th>{metric}</th><td>{value:.6f}</td></tr>")
    return "\n".join(rows)

def create_html_report(eval_results, plots_dir, output_file):
    """Create HTML report using string formatting"""
    # Load plots
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    plots = {os.path.splitext(f)[0]: encode_image(os.path.join(plots_dir, f)) 
            for f in plot_files}
    
    # Generate additional plots
    scatter_path, error_dist_path = generate_prediction_plots(
        eval_results, plots_dir)
    plots['predictions_scatter'] = encode_image(scatter_path)
    plots['error_distribution'] = encode_image(error_dist_path)
    
    # Create metrics table
    metrics_table = create_metrics_table(eval_results)
    
    # HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAN Model Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .metrics {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>HAN Model Performance Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Training Performance</h2>
            <div class="plot">
                <h3>Loss Curves</h3>
                <img src="{plots['loss_curves']}" alt="Loss Curves">
            </div>
            <div class="plot">
                <h3>Training Time</h3>
                <img src="{plots['training_time']}" alt="Training Time">
            </div>
            <div class="plot">
                <h3>Loss Distribution</h3>
                <img src="{plots['loss_distribution']}" alt="Loss Distribution">
            </div>
            <div class="plot">
                <h3>Convergence Analysis</h3>
                <img src="{plots['convergence_analysis']}" alt="Convergence Analysis">
            </div>
        </div>
        
        <div class="section">
            <h2>Evaluation Results</h2>
            <div class="metrics">
                <table>
                    {metrics_table}
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>Prediction Analysis</h2>
            <div class="plot">
                <h3>Predictions vs True Values</h3>
                <img src="{plots['predictions_scatter']}" alt="Predictions Scatter">
            </div>
            <div class="plot">
                <h3>Error Distribution</h3>
                <img src="{plots['error_distribution']}" alt="Error Distribution">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(html)

@click.command()
@click.option('--eval-results', required=True, help='Path to evaluation results JSON file')
@click.option('--plots-dir', required=True, help='Directory containing performance plots')
@click.option('--output-file', required=True, help='Output HTML report file')
def main(eval_results, plots_dir, output_file):
    """Generate comprehensive HTML report for HAN model results."""
    # Load evaluation results
    results = load_evaluation_results(eval_results)
    
    # Generate report
    create_html_report(results, plots_dir, output_file)
    print(f"Report generated: {output_file}")

if __name__ == '__main__':
    main() 