"""
Utility functions for domain classification project
"""

import json
import os
import random
from typing import List, Dict, Any


def load_jsonl_sample(filepath: str, sample_size: int = None, max_lines: int = None) -> List[Dict[str, Any]]:
    """
    Load a sample from a JSONL file
    
    Args:
        filepath: Path to JSONL file
        sample_size: Number of random samples to take (None = all)
        max_lines: Maximum lines to read before sampling (for large files)
    
    Returns:
        List of parsed JSON objects
    """
    with open(filepath, 'r') as f:
        if max_lines:
            # Read first N lines then sample
            lines = [next(f) for _ in range(min(max_lines, sum(1 for _ in f)))]
            f.seek(0)
            lines = [next(f) for _ in range(max_lines)]
        else:
            lines = f.readlines()
    
    if sample_size and sample_size < len(lines):
        lines = random.sample(lines, sample_size)
    
    return [json.loads(line) for line in lines]


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data as JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    print(f"✅ Saved to {filepath}")


def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_number(num: int) -> str:
    """Format number with commas"""
    return f"{num:,}"


def format_percentage(num: float) -> str:
    """Format percentage"""
    return f"{num*100:.1f}%"


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print progress bar"""
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    percent = 100 * current / total
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()
