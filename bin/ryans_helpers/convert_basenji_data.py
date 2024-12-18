#!/usr/bin/env python
import os
import argparse
import numpy as np
from data_processor import load_and_process_data, split_data

def main():
    parser = argparse.ArgumentParser(description='Convert Basenji TFRecords to HAN format')
    parser.add_argument('--train-pattern', required=True,
                       help='Pattern for training TFRecord files (e.g., "data/tfrecords/train-*.tfr")')
    parser.add_argument('--valid-pattern', required=True,
                       help='Pattern for validation TFRecord files')
    parser.add_argument('--test-pattern', required=True,
                       help='Pattern for test TFRecord files')
    parser.add_argument('--out-dir', required=True,
                       help='Output directory for processed data')
    parser.add_argument('--k', type=int, default=6,
                       help='k-mer size (default: 6)')
    parser.add_argument('--center-size', type=int, default=128,
                       help='Size of center window to predict (default: 128)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    train_data, kmer_to_id = load_and_process_data(
        args.train_pattern,
        k=args.k,
        center_window_size=args.center_size
    )
    
    # Process validation data
    print("Processing validation data...")
    valid_data, _ = load_and_process_data(
        args.valid_pattern,
        k=args.k,
        center_window_size=args.center_size
    )
    
    # Process test data
    print("Processing test data...")
    test_data, _ = load_and_process_data(
        args.test_pattern,
        k=args.k,
        center_window_size=args.center_size
    )
    
    # Save processed data
    print("Saving processed data...")
    
    # Save k-mer dictionary
    kmer_dict_file = os.path.join(args.out_dir, 'kmer_dict.npz')
    np.savez(kmer_dict_file, kmer_to_id=kmer_to_id)
    
    # Save training data
    train_file = os.path.join(args.out_dir, 'train_data.npz')
    np.savez(train_file,
             sequences=[item[0] for item in train_data],
             masks=[item[1] for item in train_data],
             labels=[item[2] for item in train_data])
    
    # Save validation data
    valid_file = os.path.join(args.out_dir, 'valid_data.npz')
    np.savez(valid_file,
             sequences=[item[0] for item in valid_data],
             masks=[item[1] for item in valid_data],
             labels=[item[2] for item in valid_data])
    
    # Save test data
    test_file = os.path.join(args.out_dir, 'test_data.npz')
    np.savez(test_file,
             sequences=[item[0] for item in test_data],
             masks=[item[1] for item in test_data],
             labels=[item[2] for item in test_data])
    
    print(f"Data conversion complete. Files saved in {args.out_dir}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(valid_data)}")
    print(f"Test examples: {len(test_data)}")

if __name__ == '__main__':
    main() 