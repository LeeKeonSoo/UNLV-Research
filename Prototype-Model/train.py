"""
Training script for small language models
"""

import os
import time
import torch
from contextlib import nullcontext

from models import GPTModel
from config import ModelConfig, TrainingConfig
from utils import (
    TextDataset, 
    InfiniteDataLoader, 
    get_lr, 
    estimate_loss,
    save_checkpoint,
    load_checkpoint,
    prepare_output_directory,
    count_parameters
)


def train():
    # Configuration
    model_config = ModelConfig.tiny()  # Using tiny model for MPS memory constraints
    train_config = TrainingConfig()
    
    # Print configurations
    print("=" * 80)
    print("Model Configuration:")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Heads: {model_config.n_heads}")
    print(f"  Embedding dim: {model_config.n_embd}")
    print(f"  Context length: {model_config.block_size}")
    print(f"  Estimated params: {model_config.n_params / 1e6:.1f}M")
    print()
    print("Training Configuration:")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Max steps: {train_config.max_steps}")
    print(f"  Device: {train_config.device}")
    print("=" * 80)
    
    # Setup device
    device = train_config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to mps")
        device = 'mps'

    
    # Create output directory
    prepare_output_directory(train_config.output_dir)
    
    # Load data
    print("\nLoading data...")
    try:
        train_dataset = TextDataset(train_config.data_dir, model_config.block_size, split='train')
        val_dataset = TextDataset(train_config.data_dir, model_config.block_size, split='val')
    except FileNotFoundError:
        print(f"Error: Could not find train.bin and val.bin in {train_config.data_dir}")
        print("Please prepare your data first using prepare_data.py")
        return
    
    train_loader = InfiniteDataLoader(train_dataset, train_config.batch_size, device=device)
    val_loader = InfiniteDataLoader(val_dataset, train_config.batch_size, device=device)
    
    train_iter = iter(train_loader)
    
    # Initialize model
    print("\nInitializing model...")
    model = GPTModel(model_config)
    model.to(device)
    
    # Print actual parameter count
    params = count_parameters(model)
    print(f"Actual parameters: {params['total_M']:.2f}M")
    
    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        device_type=device
    )
    
    # Compile model (PyTorch 2.0+)
    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Training context (no autocast for MPS)
    ctx = nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # Disable for MPS
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    step = 0
    running_loss = 0.0
    t0 = time.time()
    
    while step < train_config.max_steps:
        # Update learning rate
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate and save checkpoint
        if step % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters=200)
            print(f"\nStep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint_path = os.path.join(train_config.output_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, step, losses['val'], checkpoint_path)
        
        # Save regular checkpoint
        if step > 0 and step % train_config.save_interval == 0:
            checkpoint_path = os.path.join(train_config.output_dir, f'ckpt_{step}.pt')
            save_checkpoint(model, optimizer, step, running_loss / train_config.log_interval, checkpoint_path)
        
        # Training step
        for micro_step in range(train_config.gradient_accumulation_steps):
            X, Y = next(train_iter)
            
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            running_loss += loss.item() * train_config.gradient_accumulation_steps
        
        # Clip gradients
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if step % train_config.log_interval == 0 and step > 0:
            t1 = time.time()
            dt = t1 - t0
            lossf = running_loss / train_config.log_interval
            tokens_per_sec = (train_config.batch_size * train_config.gradient_accumulation_steps * 
                            model_config.block_size * train_config.log_interval) / dt
            print(f"Step {step:6d} | loss {lossf:.4f} | lr {lr:.2e} | {dt*1000/train_config.log_interval:.2f}ms/step | {tokens_per_sec/1e3:.1f}k tok/s")
            running_loss = 0.0
            t0 = time.time()
        
        step += 1
    
    # Final checkpoint
    print("\n" + "=" * 80)
    print("Training completed!")
    checkpoint_path = os.path.join(train_config.output_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, step, running_loss, checkpoint_path)


if __name__ == '__main__':
    train()
