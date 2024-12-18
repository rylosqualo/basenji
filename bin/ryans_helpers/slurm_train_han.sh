#!/bin/bash
#SBATCH --job-name=20241218_HAN_basenji
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji/%j.out
#SBATCH --error=/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ryan.keivanfar@berkeley.edu
#SBATCH --requeue

# Create output directories
OUTPUT_BASE="/clusterfs/nilah/ryank/proj/compartments/20241218_HAN_basenji"
mkdir -p "$OUTPUT_BASE"/{models,logs,results}

# Run the training script
# The script will automatically resume from the last checkpoint if it exists
python bin/ryans_helpers/train_han.py \
    --data-dir "$OUTPUT_BASE/data/processed_han" \
    --output-dir "$OUTPUT_BASE/models/han" \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 0.001 \
    --num-motif-filters 128 \
    --kmer-size 6 \
    --phrase-gru-size 64 \
    --sent-gru-size 64 \
    --num-targets 1 