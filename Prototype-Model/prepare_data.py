"""
Data preparation script - converts text files to binary format for training
"""

import os
import numpy as np
from pathlib import Path
from typing import List
import tiktoken
from datasets import load_dataset
from huggingface_hub import login as hf_login


def prepare_dataset(
    input_files: List[str],
    output_dir: str,
    train_split: float = 0.9,
    tokenizer_name: str = 'gpt2'
):
    """
    Prepare text data for training by tokenizing and splitting into train/val
    
    Args:
        input_files: list of text file paths to process
        output_dir: directory to save train.bin and val.bin
        train_split: fraction of data to use for training (rest is validation)
        tokenizer_name: tiktoken tokenizer to use (default: gpt2)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    enc = tiktoken.get_encoding(tokenizer_name)
    
    # Read and concatenate all input files
    print("Reading input files...")
    all_text = []
    for filepath in input_files:
        print(f"  - {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            all_text.append(f.read())
    
    text = '\n'.join(all_text)
    print(f"Total characters: {len(text):,}")
    
    # Tokenize
    print("Tokenizing...")
    tokens = enc.encode_ordinary(text)
    print(f"Total tokens: {len(tokens):,}")
    
    # Split into train and validation
    n = len(tokens)
    train_size = int(n * train_split)
    
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Save to binary files (using uint16 for GPT-2 vocab size < 65536)
    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)
    
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    
    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print("\nData preparation complete!")


def prepare_huggingface_dataset(dataset_name: str, output_dir: str = './data', use_auth_token: bool = True):
    """
    Prepare dataset from HuggingFace Hub
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'nampdn-ai/tiny-webtext')
        output_dir: directory to save train.bin and val.bin
        use_auth_token: whether to use HF authentication token
    """
    print(f"Loading dataset from HuggingFace: {dataset_name}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset with authentication if needed
    try:
        dataset = load_dataset(dataset_name, token=use_auth_token)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nFor gated datasets, you need to:")
        print("1. Accept the dataset terms on HuggingFace website")
        print(f"   https://huggingface.co/datasets/{dataset_name}")
        print("2. Login with: huggingface-cli login")
        print("3. Or set HF_TOKEN environment variable")
        return
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    enc = tiktoken.get_encoding('gpt2')
    
    # Process train split
    train_tokens = []
    if 'train' in dataset:
        print("Processing train split...")
        train_texts = dataset['train']['text']
        for i, text in enumerate(train_texts):
            if i % 1000 == 0:
                print(f"  Tokenizing {i}/{len(train_texts)}...")
            train_tokens.extend(enc.encode_ordinary(text))
        print(f"Train tokens: {len(train_tokens):,}")
    
    # Process validation split
    val_tokens = []
    if 'validation' in dataset:
        print("Processing validation split...")
        val_texts = dataset['validation']['text']
        for i, text in enumerate(val_texts):
            if i % 1000 == 0:
                print(f"  Tokenizing {i}/{len(val_texts)}...")
            val_tokens.extend(enc.encode_ordinary(text))
        print(f"Val tokens: {len(val_tokens):,}")
    
    # If no validation split, create one from train (90/10)
    elif 'train' in dataset and len(train_tokens) > 0:
        print("No validation split found, creating from train split (90/10)...")
        n = len(train_tokens)
        split_idx = int(n * 0.9)
        val_tokens = train_tokens[split_idx:]
        train_tokens = train_tokens[:split_idx]
        print(f"Split - Train: {len(train_tokens):,}, Val: {len(val_tokens):,}")
    
    # Save to binary files
    if len(train_tokens) > 0:
        train_arr = np.array(train_tokens, dtype=np.uint16)
        train_path = os.path.join(output_dir, 'train.bin')
        train_arr.tofile(train_path)
        print(f"Saved train.bin: {len(train_tokens):,} tokens")
    
    if len(val_tokens) > 0:
        val_arr = np.array(val_tokens, dtype=np.uint16)
        val_path = os.path.join(output_dir, 'val.bin')
        val_arr.tofile(val_path)
        print(f"Saved val.bin: {len(val_tokens):,} tokens")
    
    print("\nData preparation complete!")


def prepare_sample_data(output_dir: str = './data'):
    """
    Create a small sample dataset for testing (no auth required)
    """
    print("Creating sample dataset...")
    
    # Generate some dummy text
    sample_text = """
    The quick brown fox jumps over the lazy dog. 
    Machine learning is a subset of artificial intelligence.
    Natural language processing enables computers to understand human language.
    Deep learning models have revolutionized many fields.
    Small language models can be efficient and effective for specific tasks.
    Dataset characterization is important for understanding model behavior.
    """ * 1000  # Repeat to get more data
    
    # Save to temporary file
    temp_file = '/tmp/sample_text.txt'
    with open(temp_file, 'w') as f:
        f.write(sample_text)
    
    # Prepare dataset
    prepare_dataset([temp_file], output_dir)
    
    # Clean up
    os.remove(temp_file)


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare text data for training')
    parser.add_argument('--input', type=str, nargs='+', help='Input text files')
    parser.add_argument('--output', type=str, default='./data', help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.9, help='Training split ratio')
    parser.add_argument('--sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--hf-dataset', type=str, help='HuggingFace dataset name (e.g., nampdn-ai/tiny-webtext)')
    parser.add_argument('--no-auth', action='store_true', help='Skip HuggingFace authentication')
    parser.add_argument('--hf-token', type=str, help='HuggingFace access token')
    parser.add_argument('--hf-login', action='store_true', help='Login to HuggingFace interactively')
    
    args = parser.parse_args()
    
    # Handle HuggingFace login
    if args.hf_login:
        print("Logging in to HuggingFace...")
        hf_login()
        print("Login successful!")
    
    if args.hf_token:
        print("Using provided HuggingFace token...")
        hf_login(token=args.hf_token)
    
    if args.sample:
        prepare_sample_data(args.output)
    elif args.hf_dataset:
        prepare_huggingface_dataset(args.hf_dataset, args.output, use_auth_token=not args.no_auth)
    elif args.input:
        prepare_dataset(args.input, args.output, args.train_split)
    else:
        print("Please provide --input files, --hf-dataset, or use --sample for demo data")
        print("\nExample usage:")
        print("  # Create sample data for testing")
        print("  python prepare_data.py --sample")
        print()
        print("  # Use local text files")
        print("  python prepare_data.py --input file1.txt file2.txt --output ./data")
        print()
        print("  # Use HuggingFace dataset (requires authentication for gated datasets)")
        print("  huggingface-cli login  # First login")
        print("  python prepare_data.py --hf-dataset nampdn-ai/tiny-webtext --output ./data")
