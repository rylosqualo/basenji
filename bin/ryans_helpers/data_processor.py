import numpy as np
import itertools
from tfrecord_reader import BasenjiTFRecordReader

def generate_kmer_dict(k=6):
    """Generate a dictionary mapping k-mers to indices."""
    bases = ['A', 'C', 'G', 'T', 'N']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    kmer_to_id = {kmer: idx for idx, kmer in enumerate(kmers)}
    return kmer_to_id

def sequence_to_kmers(sequence, k=6):
    """Convert a one-hot encoded sequence to k-mer sequence."""
    # Convert one-hot to bases
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    bases = [base_map[np.argmax(base)] for base in sequence]
    seq_str = ''.join(bases)
    
    # Convert to k-mers
    kmers = [seq_str[i:i+k] for i in range(len(seq_str)-k+1)]
    return kmers

def process_sequence(sequence, doc_length=100000, sentence_length=10000, phrase_length=1000, 
                    word_length=6, kmer_to_id=None, center_window_size=128):
    """
    Process a DNA sequence into a hierarchical structure with focus on the center window.
    
    Args:
        sequence: One-hot encoded DNA sequence [length, 4]
        doc_length: Total sequence length to process
        sentence_length: Length of each sentence
        phrase_length: Length of each phrase
        word_length: Length of each word (k-mer)
        kmer_to_id: Dictionary mapping k-mers to indices
        center_window_size: Size of the center window to focus on
        
    Returns:
        tuple of (processed_sequence, center_mask)
    """
    seq_len = sequence.shape[0]
    
    # Calculate center window indices
    center_start = (seq_len - center_window_size) // 2
    center_end = center_start + center_window_size
    
    # Convert sequence to k-mers
    kmers = sequence_to_kmers(sequence, k=word_length)
    encoded_words = [kmer_to_id.get(kmer, len(kmer_to_id)) for kmer in kmers]
    
    # Create center mask for words
    word_center_start = center_start // word_length
    word_center_end = (center_end + word_length - 1) // word_length
    center_mask = np.zeros(len(encoded_words))
    center_mask[word_center_start:word_center_end] = 1
    
    # Group words into phrases
    words_per_phrase = phrase_length // word_length
    total_phrases = len(encoded_words) // words_per_phrase
    phrases = [encoded_words[i*words_per_phrase:(i+1)*words_per_phrase] for i in range(total_phrases)]
    phrase_masks = [center_mask[i*words_per_phrase:(i+1)*words_per_phrase] for i in range(total_phrases)]
    
    # Group phrases into sentences
    phrases_per_sentence = sentence_length // phrase_length
    total_sentences = len(phrases) // phrases_per_sentence
    sentences = [phrases[i*phrases_per_sentence:(i+1)*phrases_per_sentence] for i in range(total_sentences)]
    sentence_masks = [phrase_masks[i*phrases_per_sentence:(i+1)*phrases_per_sentence] for i in range(total_sentences)]
    
    return sentences, sentence_masks

def load_and_process_data(tfrecord_pattern, k=6, center_window_size=128):
    """
    Load and process genomic data from TFRecord files.
    
    Args:
        tfrecord_pattern: Pattern for TFRecord files
        k: k-mer size
        center_window_size: Size of center window to predict
        
    Returns:
        tuple of (processed_data, kmer_to_id)
    """
    # Generate k-mer dictionary
    kmer_to_id = generate_kmer_dict(k=k)
    
    # Load data from TFRecords
    reader = BasenjiTFRecordReader(tfrecord_pattern)
    sequences, labels = reader.load_data()
    
    # Process sequences and create masks
    processed_data = []
    for sequence, label in zip(sequences, labels):
        try:
            # Process sequence and get center mask
            processed_seq, center_mask = process_sequence(
                sequence,
                word_length=k,
                kmer_to_id=kmer_to_id,
                center_window_size=center_window_size
            )
            
            processed_data.append((processed_seq, center_mask, label))
            
        except Exception as e:
            print(f"Error processing sequence: {e}")
    
    return processed_data, kmer_to_id

def split_data(processed_data, train_frac=0.8, val_frac=0.1):
    """Split data into training, validation, and test sets."""
    np.random.shuffle(processed_data)
    
    n = len(processed_data)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    
    train_data = processed_data[:train_end]
    val_data = processed_data[train_end:val_end]
    test_data = processed_data[val_end:]
    
    return train_data, val_data, test_data 