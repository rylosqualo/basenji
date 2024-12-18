#!/bin/bash

# Set the base directory
BASE_DIR="/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji"

# Set directory paths
BASENJI_DATA_DIR="$BASE_DIR/testdata/cage/tfrecord"
DATA_DIR="$BASE_DIR/data/processed_han"
MODEL_DIR="$BASE_DIR/models/han"
RESULTS_DIR="$BASE_DIR/results/han"

# Create directories if they don't exist
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR
mkdir -p $RESULTS_DIR

# Launch the Nextflow pipeline
nextflow run "$BASE_DIR/bin/ryans_helpers/han_pipeline.nf" \
    -c "$BASE_DIR/bin/ryans_helpers/nextflow.config" \
    -profile slurm \
    --output_base $BASE_DIR \
    --basenji_data $BASENJI_DATA_DIR \
    -resume

# Print completion message
echo "Pipeline submission completed. Check status with 'squeue -u $USER'" 