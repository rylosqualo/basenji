#!/bin/bash

# Default values
DATA_DIR="data/processed_han"
OUTPUT_DIR="models/han"
BATCH_SIZE=32
NUM_EPOCHS=100
LEARNING_RATE=0.001
NUM_MOTIF_FILTERS=128
KMER_SIZE=6
PHRASE_GRU_SIZE=64
SENT_GRU_SIZE=64
NUM_TARGETS=1
DROPOUT=0.2
VAL_SPLIT=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-motif-filters)
            NUM_MOTIF_FILTERS="$2"
            shift 2
            ;;
        --kmer-size)
            KMER_SIZE="$2"
            shift 2
            ;;
        --phrase-gru-size)
            PHRASE_GRU_SIZE="$2"
            shift 2
            ;;
        --sent-gru-size)
            SENT_GRU_SIZE="$2"
            shift 2
            ;;
        --num-targets)
            NUM_TARGETS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --val-split)
            VAL_SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the training script
python bin/ryans_helpers/train_han.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-epochs "$NUM_EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --num-motif-filters "$NUM_MOTIF_FILTERS" \
    --kmer-size "$KMER_SIZE" \
    --phrase-gru-size "$PHRASE_GRU_SIZE" \
    --sent-gru-size "$SENT_GRU_SIZE" \
    --num-targets "$NUM_TARGETS" \
    --dropout "$DROPOUT" \
    --val-split "$VAL_SPLIT"

echo "Training completed. Model saved to $OUTPUT_DIR" 