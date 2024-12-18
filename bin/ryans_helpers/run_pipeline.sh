#!/bin/bash

# Set the base directory
BASE_DIR="/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji"

# Set the output directory
OUTPUT_DIR="$BASE_DIR/results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Launch the Nextflow pipeline
nextflow run "$BASE_DIR/bin/ryans_helpers/han_pipeline.nf" \
    -c "$BASE_DIR/bin/ryans_helpers/nextflow.config" \
    -profile slurm \
    --output_base $OUTPUT_DIR \
    -resume

# Print completion message
echo "Pipeline submission completed. Check status with 'squeue -u $USER'" 