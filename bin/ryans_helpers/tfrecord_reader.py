#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
from glob import glob

class BasenjiTFRecordReader:
    """Reader for Basenji TFRecord files."""
    
    def __init__(self, tfrecord_pattern):
        """Initialize the reader.
        
        Args:
            tfrecord_pattern: Pattern for TFRecord files (e.g., "data/tfrecords/train-*.tfr")
        """
        self.tfrecord_files = sorted(glob(tfrecord_pattern))
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found matching pattern: {tfrecord_pattern}")
    
    def parse_example(self, example_proto):
        """Parse a single TFRecord example."""
        feature_description = {
            'sequence': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string)
        }
        
        # Parse the example
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode sequence (uint8)
        sequence = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
        sequence = tf.reshape(sequence, [-1, 4])  # reshape to [length, 4]
        
        # Decode targets (float16)
        targets = tf.io.decode_raw(parsed_features['target'], tf.float16)
        targets = tf.cast(targets, tf.float32)  # convert to float32
        targets = tf.reshape(targets, [-1, -1])  # reshape to [length, num_targets]
        
        return sequence, targets

    def dataset_iterator(self, batch_size=1):
        """Create a TensorFlow dataset iterator.
        
        Args:
            batch_size: Batch size for the dataset
            
        Returns:
            A TensorFlow dataset that yields (sequence, targets) tuples
        """
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(self.tfrecord_files, compression_type='ZLIB')
        
        # Parse examples
        dataset = dataset.map(self.parse_example)
        
        # Batch the data
        dataset = dataset.batch(batch_size)
        
        return dataset

    def extract_center_values(self, sequence, targets, center_size=128):
        """Extract the center window from sequence and corresponding target values.
        
        Args:
            sequence: One-hot encoded DNA sequence [length, 4]
            targets: Target values [length, num_targets]
            center_size: Size of the center window to extract
            
        Returns:
            tuple of (full_sequence, center_target_values)
        """
        # Get sequence length
        seq_len = sequence.shape[0]
        
        # Calculate center indices
        center_start = (seq_len - center_size) // 2
        center_end = center_start + center_size
        
        # Extract center target values
        center_targets = targets[center_start:center_end]
        
        # Calculate mean target values for the center window
        center_target_values = tf.reduce_mean(center_targets, axis=0)
        
        return sequence, center_target_values

    def load_data(self, max_examples=None):
        """Load data from TFRecord files.
        
        Args:
            max_examples: Maximum number of examples to load (None for all)
            
        Returns:
            tuple of (sequences, center_target_values)
        """
        sequences = []
        center_values = []
        
        dataset = self.dataset_iterator()
        
        for i, (seq, tgt) in enumerate(dataset):
            if max_examples and i >= max_examples:
                break
                
            # Process each example in the batch
            for j in range(seq.shape[0]):
                sequence = seq[j].numpy()
                targets = tgt[j].numpy()
                
                # Extract center values
                _, center_target = self.extract_center_values(sequence, targets)
                
                sequences.append(sequence)
                center_values.append(center_target.numpy())
        
        return np.array(sequences), np.array(center_values)

def convert_basenji_to_han(tfrecord_pattern, output_file=None, max_examples=None):
    """Convert Basenji TFRecord data to format suitable for HAN model.
    
    Args:
        tfrecord_pattern: Pattern for TFRecord files
        output_file: Optional output file to save the converted data
        max_examples: Maximum number of examples to convert
        
    Returns:
        tuple of (sequences, labels) suitable for HAN model
    """
    # Load data from TFRecords
    reader = BasenjiTFRecordReader(tfrecord_pattern)
    sequences, center_values = reader.load_data(max_examples)
    
    if output_file:
        np.savez(output_file, 
                 sequences=sequences,
                 labels=center_values)
    
    return sequences, center_values

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Basenji TFRecords to HAN format')
    parser.add_argument('tfrecord_pattern', help='Pattern for TFRecord files')
    parser.add_argument('--output', help='Output file (.npz)')
    parser.add_argument('--max-examples', type=int, help='Maximum examples to convert')
    
    args = parser.parse_args()
    
    convert_basenji_to_han(args.tfrecord_pattern, 
                          args.output,
                          args.max_examples) 