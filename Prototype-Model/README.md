# Small Language Model Training Framework

A lightweight framework for training and experimenting with Small Language Models (SLMs) for dataset characterization research.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy tiktoken
```

### 2. Prepare Data

Option A - Use sample data for testing:
```bash
python prepare_data.py --sample
```

Option B - Use your own text files:
```bash
python prepare_data.py --input data1.txt data2.txt --output ./data
```

### 3. Train Model

```bash
python train.py
```

The script will:
- Load data from `./data/` directory
- Train a ~100M parameter GPT-style model
- Save checkpoints to `./checkpoints/`

### 4. Generate Text

```bash
# Single generation
python generate.py --checkpoint ./checkpoints/best_model.pt --prompt "Once upon a time"

# Interactive mode
python generate.py --checkpoint ./checkpoints/best_model.pt --interactive
```

## Model Sizes

You can adjust model size in `train.py` by changing:

```python
model_config = ModelConfig.tiny()    # ~50M parameters
model_config = ModelConfig.small()   # ~100M parameters
model_config = ModelConfig.medium()  # ~300M parameters
```

## File Structure

```
.
├── config.py          # Model and training configurations
├── models.py          # GPT-style transformer implementation
├── utils.py           # Data loading and training utilities
├── train.py           # Training script
├── generate.py        # Text generation script
├── prepare_data.py    # Data preprocessing script
├── data/              # Training data (train.bin, val.bin)
└── checkpoints/       # Model checkpoints
```

## Key Features

- **Efficient Training**: Infinite data loader with random sampling
- **Mixed Precision**: Automatic mixed precision training (bfloat16)
- **Gradient Accumulation**: Support for larger effective batch sizes
- **Cosine LR Schedule**: Linear warmup + cosine decay
- **Weight Decay**: Proper parameter grouping (decay only on 2D weights)
- **Checkpointing**: Regular and best-model checkpoints

## Training Configuration

Key hyperparameters in `config.py`:

```python
batch_size = 32                    # Per-device batch size
gradient_accumulation_steps = 4    # Effective batch = 128
learning_rate = 3e-4              # Peak learning rate
warmup_steps = 2000               # LR warmup steps
max_steps = 100000                # Total training steps
```

## Usage for Dataset Experiments

This framework is designed for testing different dataset compositions:

1. Prepare multiple datasets with different characteristics
2. Train prototype models (100M-300M params) on each
3. Evaluate on downstream tasks
4. Compare performance to identify best dataset mixtures

Example workflow:
```bash
# Prepare different dataset variants
python prepare_data.py --input dataset_v1/*.txt --output ./data_v1
python prepare_data.py --input dataset_v2/*.txt --output ./data_v2

# Train models on each variant
# (modify data_dir in config.py for each run)
python train.py  # using data_v1
python train.py  # using data_v2

# Compare results
```

## Performance Tips

- Use CUDA if available (20-50x faster than CPU)
- Increase `batch_size` if you have more GPU memory
- Use `compile=True` in config for PyTorch 2.0+ (10-20% speedup)
- Adjust `block_size` based on your data characteristics

## Notes

- Data should be pre-tokenized as `.bin` files (uint16 format)
- Uses GPT-2 tokenizer (tiktoken) by default
- Weight tying between token embeddings and LM head
- Causal (autoregressive) attention mask
