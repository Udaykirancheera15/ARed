#!/usr/bin/env python3
"""Shared utilities for AUTORED framework"""

import os
import json
import torch
import numpy as np
import random
from typing import Dict, List, Optional

# ============================================================
# SAFETY UTILITIES
# ============================================================

def safe_tensor(tensor, nan_val=0.0, inf_val=1e4):
    """Replace NaN and Inf values in tensors"""
    if tensor is None:
        return None
    tensor = torch.nan_to_num(tensor, nan=nan_val, posinf=inf_val, neginf=-inf_val)
    return tensor

def safe_probs(logits, temperature=1.0):
    """Compute safe probabilities from logits"""
    logits = safe_tensor(logits.float())
    logits = torch.clamp(logits, min=-50, max=50)
    logits = logits / temperature
    
    # Block bad tokens
    bad_ids = [0, 1, 2, 3]  # pad, eos, unk, etc.
    for bid in bad_ids:
        if bid < logits.size(-1):
            logits[..., bid] = -1e9
    
    probs = torch.softmax(logits, dim=-1)
    probs = safe_tensor(probs, nan_val=1e-8)
    
    # Ensure valid distribution
    probs = torch.clamp(probs, min=1e-10)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return probs

def set_seed(seed=42):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_jsonl(path):
    """Load JSONL file"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    """Save to JSONL file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
