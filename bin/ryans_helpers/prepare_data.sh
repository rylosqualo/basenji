#!/bin/bash

# Default values
BASENJI_DATA_DIR="data/train"
OUTPUT_DIR="data/processed_han"
NUM_PROCESSES=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --basenji-data)
            BASENJI_DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-processes)
            NUM_PROCESSES="$2"
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

# Run the data preparation script
python bin/ryans_helpers/prepare_han_data.py \
    --basenji-data "$BASENJI_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-processes "$NUM_PROCESSES"

echo "Data preparation completed. Output saved to $OUTPUT_DIR" 