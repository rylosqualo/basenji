# HAN Model for Genomic Sequence Analysis

## Overview

This pipeline implements a Hierarchical Attention Network (HAN) adapted for genomic sequence analysis. The model predicts regulatory activity values for the centermost 128bp window of a 131,072bp sequence, using the entire sequence context for prediction.

## Model Architecture

The HAN model consists of several hierarchical levels:

1. **Motif Detection Layer**
   - Uses a CNN with a kernel size of 6 (k-mer size)
   - Learns motif representations directly from raw sequences
   - Input: One-hot encoded DNA sequence (4 x 131,072)
   - Output: Learned motif features (num_filters x sequence_length)

2. **Phrase-Level Processing**
   - Groups nearby motifs into phrases (128bp windows)
   - Uses bidirectional GRU to process each phrase
   - Attention mechanism to weight important motifs within phrases
   - Captures local sequence patterns

3. **Sentence-Level Processing**
   - Groups phrases into larger contexts
   - Another bidirectional GRU with attention
   - Captures medium-range interactions
   - Helps understand broader sequence context

4. **Document-Level Integration**
   - Final attention mechanism over all sentence representations
   - Creates a single vector representing the entire sequence
   - Maintains global context while focusing on relevant regions

5. **Prediction Layer**
   - Multi-layer perceptron for final prediction
   - Predicts regulatory activity values for center window
   - Uses global context to inform local predictions

## Labels

The model is trained on regulatory activity values from Basenji's TFRecord format:

1. **Input Sequences**
   - 131,072bp genomic sequences
   - One-hot encoded (A,C,G,T)
   - Centered on regions of interest

2. **Target Values**
   - Regulatory activity measurements for center 128bp
   - Continuous values representing activity levels
   - Examples: chromatin accessibility, transcription factor binding, etc.
   - Normalized to zero mean and unit variance

## Training Process

1. **Data Preparation**
   - Converts Basenji TFRecords to HAN format
   - Splits data into train/val/test sets
   - Handles sequence padding and normalization

2. **Training**
   - Uses MSE loss for regression
   - Trains on full sequences but predicts center window
   - Regular checkpointing for interruption recovery
   - Monitors validation loss for early stopping

3. **Evaluation**
   - Comprehensive metrics (MSE, MAE, R²)
   - Performance visualization
   - Prediction analysis and error distribution

## Pipeline Components

1. `train_han.py`: Main training script
2. `evaluate_han.py`: Model evaluation
3. `plot_performance.py`: Performance visualization
4. `generate_report.py`: HTML report generation
5. `han_pipeline.nf`: Nextflow workflow orchestration

## Usage

Run the complete pipeline:
```bash
nextflow run han_pipeline.nf \
    --output_base /path/to/output \
    --data_dir /path/to/data
```

The pipeline will:
1. Train the model
2. Generate performance plots
3. Evaluate on test set
4. Create comprehensive report

## Dependencies

- Python 3.8+
- PyTorch
- Nextflow
- Matplotlib
- Seaborn
- Click
- Jinja2

## Output Structure

```
output_base/
├── data/
│   └── processed_han/
├── models/
│   └── han/
│       ├── checkpoints/
│       ├── metrics.jsonl
│       └── TRAINING_COMPLETED
├── results/
│   └── han/
│       ├── plots/
│       ├── evaluation_results.json
│       └── report.html
└── logs/
``` 