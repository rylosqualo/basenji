#!/bin/bash

# Set paths
REPO_DIR="/clusterfs/nilah/ryank/proj/compartments/basenji"  # Source code and input data
WORK_DIR="/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji"  # All outputs

# Create output directories
mkdir -p "$WORK_DIR/data/processed_han"
mkdir -p "$WORK_DIR/models/han"
mkdir -p "$WORK_DIR/results/han"
mkdir -p "$WORK_DIR/pipeline_info"
mkdir -p "$WORK_DIR/nextflow_logs"

# Set Nextflow home to work directory to contain all Nextflow-related files
export NXF_HOME="$WORK_DIR/nextflow_logs"

# Change to work directory
cd $WORK_DIR

# Launch the Nextflow pipeline
nextflow run "$REPO_DIR/bin/ryans_helpers/han_pipeline.nf" \
    -c "$REPO_DIR/bin/ryans_helpers/nextflow.config" \
    -profile slurm \
    --repo_dir $REPO_DIR \
    --output_base $WORK_DIR \
    -resume \
    -with-trace "$WORK_DIR/pipeline_info/trace.txt" \
    -with-timeline "$WORK_DIR/pipeline_info/timeline.html" \
    -with-report "$WORK_DIR/pipeline_info/report.html" \
    -with-dag "$WORK_DIR/pipeline_info/dag.html"

# Check if pipeline submission was successful
if [ $? -eq 0 ]; then
    echo "Pipeline successfully submitted. Check status with 'squeue -u $USER'"
else
    echo "Pipeline submission failed. Check logs in $WORK_DIR/nextflow_logs/.nextflow.log"
fi