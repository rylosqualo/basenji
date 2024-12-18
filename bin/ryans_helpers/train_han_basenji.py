#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from HAN import HAN
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class BasenjiDataset(Dataset):
    """Dataset for loading processed Basenji data."""
    def __init__(self, data_file):
        data = np.load(data_file)
        self.sequences = data['sequences']
        self.masks = data['masks']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], self.masks[idx], self.labels[idx])

def collate_fn(batch):
    """Custom collate function for batching data."""
    sequences = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Find max lengths
    max_sent_len = max(len(doc) for doc in sequences)
    max_phrase_len = max(len(sent) for doc in sequences for sent in doc)
    max_word_len = max(len(phrase) for doc in sequences for sent in doc for phrase in sent)
    
    # Pad sequences and masks
    padded_sequences = []
    padded_masks = []
    
    for seq, mask in zip(sequences, masks):
        # Pad sentences
        padded_seq = []
        padded_mask = []
        
        for sent, sent_mask in zip(seq, mask):
            # Pad phrases
            padded_sent = []
            padded_sent_mask = []
            
            for phrase, phrase_mask in zip(sent, sent_mask):
                # Pad words
                num_words = len(phrase)
                padded_phrase = phrase + [0] * (max_word_len - num_words)
                padded_phrase_mask = np.concatenate([phrase_mask, np.zeros(max_word_len - num_words)])
                
                padded_sent.append(padded_phrase)
                padded_sent_mask.append(padded_phrase_mask)
            
            # Pad phrases in sentence
            num_phrases = len(padded_sent)
            for _ in range(max_phrase_len - num_phrases):
                padded_sent.append([0] * max_word_len)
                padded_sent_mask.append(np.zeros(max_word_len))
            
            padded_seq.append(padded_sent)
            padded_mask.append(padded_sent_mask)
        
        # Pad sentences in document
        num_sentences = len(padded_seq)
        for _ in range(max_sent_len - num_sentences):
            padded_seq.append([[0] * max_word_len] * max_phrase_len)
            padded_mask.append(np.zeros((max_phrase_len, max_word_len)))
        
        padded_sequences.append(padded_seq)
        padded_masks.append(padded_mask)
    
    # Convert to tensors
    sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    masks_tensor = torch.tensor(padded_masks, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    
    return sequences_tensor, masks_tensor, labels_tensor

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, checkpoint_dir):
    """Train the model and save checkpoints."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            sequences, masks, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(sequences, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, masks, labels = [x.to(device) for x in batch]
                outputs = model(sequences, masks)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate metrics
        val_r2 = r2_score(val_true, val_preds)
        val_mse = mean_squared_error(val_true, val_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val R2: {val_r2:.4f}")
        print(f"Val MSE: {val_mse:.4f}")
        
        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))
    plt.close()
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='Train HAN model on Basenji data')
    parser.add_argument('--data-dir', required=True,
                       help='Directory containing processed data files')
    parser.add_argument('--out-dir', required=True,
                       help='Output directory for checkpoints and results')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--word-hidden', type=int, default=50,
                       help='Word-level GRU hidden size')
    parser.add_argument('--phrase-hidden', type=int, default=50,
                       help='Phrase-level GRU hidden size')
    parser.add_argument('--sent-hidden', type=int, default=50,
                       help='Sentence-level GRU hidden size')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load k-mer dictionary
    kmer_dict = np.load(os.path.join(args.data_dir, 'kmer_dict.npz'))
    vocab_size = len(kmer_dict['kmer_to_id']) + 1  # +1 for padding
    
    # Create datasets
    train_dataset = BasenjiDataset(os.path.join(args.data_dir, 'train_data.npz'))
    valid_dataset = BasenjiDataset(os.path.join(args.data_dir, 'valid_data.npz'))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = HAN(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        word_gru_hidden_size=args.word_hidden,
        phrase_gru_hidden_size=args.phrase_hidden,
        sent_gru_hidden_size=args.sent_hidden
    ).to(device)
    
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.out_dir
    )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_final.pt'))
    
    print("Training complete!")

if __name__ == '__main__':
    main() 