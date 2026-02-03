"""
Test script to debug the weight tying issue
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/Users/leekeonsoo/Desktop/University/Etc./UNLV/Lab/Research')

from models import GPTModel
from config import ModelConfig

config = ModelConfig.small()
model = GPTModel(config)

print("\n" + "="*80)
print("ALL NAMED PARAMETERS:")
print("="*80)
all_params = {}
for name, param in model.named_parameters():
    all_params[name] = param.shape
    print(f"  {name}: {param.shape}")

print("\n" + "="*80)
print("WEIGHT TYING CHECK:")
print("="*80)
print(f"transformer.wte.weight is lm_head.weight: {model.transformer.wte.weight is model.lm_head.weight}")
print(f"transformer.wte.weight id: {id(model.transformer.wte.weight)}")
print(f"lm_head.weight id: {id(model.lm_head.weight)}")

print("\n" + "="*80)
print("CHECKING OPTIMIZER PARAMETER GROUPING:")
print("="*80)

# Run the same logic as configure_optimizers
decay = set()
no_decay = set()
whitelist_weight_modules = (torch.nn.Linear, )
blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

for mn, m in model.named_modules():
    for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn
        
        if pn.endswith('bias'):
            no_decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
            decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
            no_decay.add(fpn)

param_dict = {pn: p for pn, p in model.named_parameters()}
union_params = decay | no_decay
missing_params = param_dict.keys() - union_params

print(f"\nDecay parameters ({len(decay)}):")
for p in sorted(decay):
    print(f"  {p}")

print(f"\nNo-decay parameters ({len(no_decay)}):")
for p in sorted(no_decay):
    print(f"  {p}")

print(f"\nMissing parameters ({len(missing_params)}):")
for p in sorted(missing_params):
    print(f"  {p}")

if len(missing_params) > 0:
    print("\nThese missing parameters will be added to no_decay")
    
print("\n" + "="*80)
print("TESTING OPTIMIZER CREATION:")
print("="*80)

# Add missing params to no_decay
if len(missing_params) > 0:
    no_decay.update(missing_params)

# Try creating optimizer groups
try:
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    print("✓ Optimizer groups created successfully!")
    print(f"  Group 1 (decay): {len(decay)} parameters")
    print(f"  Group 2 (no_decay): {len(no_decay)} parameters")
except KeyError as e:
    print(f"✗ KeyError: {e}")
