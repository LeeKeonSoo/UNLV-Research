"""
Test script to debug the weight tying issue - Part 2
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
print("CHECKING NAMED_MODULES vs NAMED_PARAMETERS:")
print("="*80)

# Collect parameters from named_modules
params_from_modules = set()
for mn, m in model.named_modules():
    for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn
        params_from_modules.add(fpn)
        
# Collect parameters from named_parameters
params_from_named = set()
for pn, p in model.named_parameters():
    params_from_named.add(pn)

print(f"Parameters from named_modules: {len(params_from_modules)}")
print(f"Parameters from named_parameters: {len(params_from_named)}")

print("\nIn named_modules but NOT in named_parameters:")
for p in sorted(params_from_modules - params_from_named):
    print(f"  {p}")

print("\nIn named_parameters but NOT in named_modules:")
for p in sorted(params_from_named - params_from_modules):
    print(f"  {p}")

print("\n" + "="*80)
print("CHECKING lm_head specifically:")
print("="*80)

# Check if lm_head appears in named_modules
lm_head_in_modules = False
for mn, m in model.named_modules():
    if mn == 'lm_head':
        lm_head_in_modules = True
        print(f"Found lm_head in named_modules")
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            print(f"  Parameter: {fpn}")

# Check if lm_head.weight is in named_parameters
lm_head_in_params = False
for pn, p in model.named_parameters():
    if pn == 'lm_head.weight':
        lm_head_in_params = True
        print(f"Found lm_head.weight in named_parameters()")
        
if not lm_head_in_params:
    print("lm_head.weight NOT found in named_parameters()")
    
print(f"\nDirect check:")
print(f"  hasattr(model.lm_head, 'weight'): {hasattr(model.lm_head, 'weight')}")
print(f"  model.lm_head.weight shape: {model.lm_head.weight.shape}")
print(f"  id(model.lm_head.weight): {id(model.lm_head.weight)}")
print(f"  id(model.transformer.wte.weight): {id(model.transformer.wte.weight)}")
print(f"  Same object? {model.lm_head.weight is model.transformer.wte.weight}")
