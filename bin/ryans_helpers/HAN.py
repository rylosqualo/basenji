# Import necessary libraries
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MotifModule(nn.Module):
    """
    CNN-based module for learning motif representations from raw sequences.
    Uses sliding window convolution to detect k-mer motifs.
    """
    def __init__(self, num_filters=128, kmer_size=6):
        super(MotifModule, self).__init__()
        
        # Single convolutional layer for k-mer motif detection
        self.conv = nn.Conv1d(4, num_filters, kernel_size=kmer_size, padding='same')
        self.batch_norm = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: One-hot encoded sequences (batch_size, 4, seq_length)
            
        Returns:
            Learned motif representations (batch_size, num_filters, seq_length)
        """
        # Apply convolution to detect k-mer motifs
        motif_features = self.conv(x)
        motif_features = self.batch_norm(motif_features)
        motif_features = self.relu(motif_features)
        
        return motif_features

class HAN(nn.Module):
    """
    Hierarchical Attention Network adapted for genomic sequences.
    Uses entire sequence context to predict values for the center window.
    
    Architecture:
    - Motif detection using CNN (k=6, sliding window)
    - Phrase-level encoder and attention (subsequences)
    - Sentence-level encoder and attention (larger subsequences)
    - Document-level representation (full sequence)
    - Final prediction for center window (128 bp)
    """
    def __init__(self, num_motif_filters=128, kmer_size=6, 
                 phrase_gru_hidden_size=64, sent_gru_hidden_size=64, 
                 num_targets=1):
        super(HAN, self).__init__()

        # Motif detection module
        self.motif_module = MotifModule(
            num_filters=num_motif_filters,
            kmer_size=kmer_size
        )

        # Phrase-level GRU and attention
        self.phrase_gru = nn.GRU(num_motif_filters, phrase_gru_hidden_size, 
                                bidirectional=True, batch_first=True)
        self.phrase_attention = nn.Linear(phrase_gru_hidden_size * 2, phrase_gru_hidden_size * 2)
        self.phrase_context_vector = nn.Linear(phrase_gru_hidden_size * 2, 1, bias=False)

        # Sentence-level GRU and attention
        self.sent_gru = nn.GRU(phrase_gru_hidden_size * 2, sent_gru_hidden_size, 
                              bidirectional=True, batch_first=True)
        self.sent_attention = nn.Linear(sent_gru_hidden_size * 2, sent_gru_hidden_size * 2)
        self.sent_context_vector = nn.Linear(sent_gru_hidden_size * 2, 1, bias=False)

        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(sent_gru_hidden_size * 2, sent_gru_hidden_size),
            nn.ReLU(),
            nn.Linear(sent_gru_hidden_size, sent_gru_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(sent_gru_hidden_size // 2, num_targets)
        )

    def forward(self, x):
        """
        Process the entire sequence and predict values for the center window.
        The prediction is based on information from the entire sequence.
        
        Args:
            x: One-hot encoded sequences (batch_size, seq_length, 4)
            
        Returns:
            predictions for the center window based on full sequence context
        """
        batch_size = x.size(0)
        
        # Transpose for conv1d (batch_size, channels, seq_length)
        x = x.transpose(1, 2)
        
        # Detect motifs using CNN
        motif_features = self.motif_module(x)  # (batch_size, num_filters, seq_length)
        
        # Reshape motif features into phrases
        # We'll use a sliding window approach to group nearby motifs
        phrase_length = 128  # Size of each phrase in bp
        stride = phrase_length // 2  # 50% overlap between phrases
        
        phrases = []
        for i in range(0, motif_features.size(2) - phrase_length + 1, stride):
            phrase = motif_features[:, :, i:i+phrase_length]
            phrases.append(phrase)
        
        # Stack phrases
        phrases = torch.stack(phrases, dim=1)  # (batch_size, num_phrases, num_filters, phrase_length)
        phrases = phrases.transpose(2, 3)  # (batch_size, num_phrases, phrase_length, num_filters)
        
        # Process each phrase
        batch_size, num_phrases, phrase_length, num_filters = phrases.shape
        phrases = phrases.reshape(-1, phrase_length, num_filters)
        
        # Phrase-level GRU
        phrase_gru_out, _ = self.phrase_gru(phrases)

        # Phrase-level attention
        phrase_att = torch.tanh(self.phrase_attention(phrase_gru_out))
        phrase_att = self.phrase_context_vector(phrase_att).squeeze(-1)
        phrase_att = F.softmax(phrase_att, dim=1)
        phrase_att = phrase_att.unsqueeze(-1)
        sent_embeddings = torch.sum(phrase_att * phrase_gru_out, dim=1)

        # Reshape for sentence-level processing
        sent_embeddings = sent_embeddings.view(batch_size, num_phrases, -1)

        # Sentence-level GRU
        sent_gru_out, _ = self.sent_gru(sent_embeddings)

        # Sentence-level attention
        sent_att = torch.tanh(self.sent_attention(sent_gru_out))
        sent_att = self.sent_context_vector(sent_att).squeeze(-1)
        sent_att = F.softmax(sent_att, dim=1)
        sent_att = sent_att.unsqueeze(-1)
        
        # Get sequence representation that captures global context
        sequence_repr = torch.sum(sent_att * sent_gru_out, dim=1)
        
        # Generate predictions for center window using global context
        output = self.final_layers(sequence_repr)
        
        return output

def process_sequence(seq, seq_length=131072):
    """
    Convert DNA sequence to one-hot encoding.
    
    Args:
        seq: DNA sequence string
        seq_length: Target sequence length
        
    Returns:
        One-hot encoded sequence (4 x seq_length)
    """
    # Ensure sequence is the correct length
    if len(seq) < seq_length:
        # Pad with N's if too short
        seq = seq + 'N' * (seq_length - len(seq))
    elif len(seq) > seq_length:
        # Trim from both ends to keep center if too long
        trim_amount = (len(seq) - seq_length) // 2
        seq = seq[trim_amount:trim_amount + seq_length]
    
    # Define the mapping
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Create one-hot encoding
    one_hot = np.zeros((4, seq_length), dtype=np.float32)
    
    for i, base in enumerate(seq):
        if base in base_to_idx:
            idx = base_to_idx[base]
            if idx < 4:  # Only one-hot encode ACGT, treat N as all zeros
                one_hot[idx, i] = 1
    
    return one_hot