"""
Utility functions for data loading, tokenization, and training helpers
"""

import os
import torch
import numpy as np
from typing import Iterator, Tuple
from pathlib import Path


class TextDataset:
    """Simple text dataset that loads pre-tokenized data"""
    
    def __init__(self, data_path: str, block_size: int, split: str = 'train'):
        """
        Args:
            data_path: path to directory containing train.bin and val.bin
            block_size: context length
            split: 'train' or 'val'
        """
        self.block_size = block_size
        
        data_file = os.path.join(data_path, f'{split}.bin')
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        
        print(f"Loaded {split} data: {len(self.data):,} tokens")
    
    def __len__(self):
        # Return number of possible blocks
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a block of tokens starting at idx
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


class InfiniteDataLoader:
    """
    Infinite data loader that samples random blocks from the dataset
    More efficient than standard PyTorch DataLoader for language modeling
    """
    
    def __init__(self, dataset: TextDataset, batch_size: int, device: str = 'cuda'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            # Sample random starting positions
            ix = torch.randint(len(self.dataset), (self.batch_size,))
            
            # Gather batches
            x = torch.stack([self.dataset[i][0] for i in ix])
            y = torch.stack([self.dataset[i][1] for i in ix])
            
            # Transfer to device
            # Note: pin_memory() doesn't work with MPS, only use it for CUDA
            if self.device == 'cuda':
                x = x.pin_memory().to(self.device, non_blocking=True)
                y = y.pin_memory().to(self.device, non_blocking=True)
            else:
                x, y = x.to(self.device), y.to(self.device)
            
            yield x, y


def get_lr(step: int, config) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay
    
    Args:
        step: current training step
        config: TrainingConfig object
    """
    warmup_steps = config.warmup_steps
    max_steps = config.lr_decay_steps if config.lr_decay_steps else config.max_steps
    learning_rate = config.learning_rate
    min_lr = config.min_lr
    
    # Linear warmup
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    
    # Cosine decay
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters: int = 200):
    """
    Estimate loss on train and validation sets
    
    Args:
        model: the model to evaluate
        train_loader: training data loader
        val_loader: validation data loader
        eval_iters: number of iterations to average over
    """
    out = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        loader_iter = iter(loader)
        
        for k in range(eval_iters):
            X, Y = next(loader_iter)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
    return out


def save_checkpoint(model, optimizer, step, loss, path: str):
    """Save model checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'loss': loss,
        'config': model.config,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', None)
    
    print(f"Loaded checkpoint from {path} (step {step})")
    return step, loss


def prepare_data_directory(data_dir: str):
    """Create data directory if it doesn't exist"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


def prepare_output_directory(output_dir: str):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6
    }


# Simple tokenizer wrapper for quick testing
class SimpleTokenizer:
    """
    Minimal character-level tokenizer for testing
    For real experiments, use tiktoken or HuggingFace tokenizers
    """
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
    
    def encode(self, text):
        # Simple byte-level encoding
        return list(text.encode('utf-8'))
    
    def decode(self, tokens):
        # Simple byte-level decoding
        return bytes(tokens).decode('utf-8', errors='ignore')
