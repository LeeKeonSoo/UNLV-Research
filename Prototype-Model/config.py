"""
Configuration for Small Language Model training
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Model size presets: tiny (~50M), small (~100M), medium (~300M)
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    block_size: int = 1024  # context length
    dropout: float = 0.1
    bias: bool = True  # use bias in linear layers
    
    @property
    def n_params(self) -> int:
        """Estimate number of parameters"""
        # Rough estimate: embedding + transformer blocks + final layer norm + lm_head
        embed_params = self.vocab_size * self.n_embd  # token embeddings
        pos_params = self.block_size * self.n_embd  # position embeddings
        
        # Per transformer block
        attn_params = 4 * self.n_embd * self.n_embd  # Q, K, V, proj
        mlp_params = 8 * self.n_embd * self.n_embd  # 2 linear layers (4x expansion)
        block_params = (attn_params + mlp_params) * self.n_layers
        
        # LM head (shares weights with token embedding usually, but count it)
        lm_head_params = self.vocab_size * self.n_embd
        
        total = embed_params + pos_params + block_params + lm_head_params
        return total
    
    @staticmethod
    def tiny():
        """~50M parameters"""
        return ModelConfig(
            n_layers=6,
            n_heads=8,
            n_embd=512,
            block_size=512
        )
    
    @staticmethod
    def small():
        """~100M parameters"""
        return ModelConfig(
            n_layers=12,
            n_heads=12,
            n_embd=768,
            block_size=1024
        )
    
    @staticmethod
    def medium():
        """~300M parameters"""
        return ModelConfig(
            n_layers=24,
            n_heads=16,
            n_embd=1024,
            block_size=1024
        )


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    batch_size: int = 4  # Reduced for MPS memory
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size of 32
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 2000
    max_steps: int = 100000
    lr_decay_steps: Optional[int] = None  # if None, use max_steps
    min_lr: float = 3e-5  # minimum learning rate (1/10 of max)
    
    # Checkpointing
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    
    # System
    device: str = 'cuda'
    compile: bool = False  # PyTorch 2.0 compile
    num_workers: int = 4
    
    # Paths
    data_dir: str = './data'
    output_dir: str = './checkpoints'
    log_dir: str = './logs'
