#!/bin/bash

# Set the output directory
OUTPUT_DIR="/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji/results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Launch the Nextflow pipeline
nextflow run main.nf \
    -profile slurm \
    --output_base $OUTPUT_DIR \
    -resume

# Print completion message
echo "Pipeline submission completed. Check status with 'squeue -u $USER'" 