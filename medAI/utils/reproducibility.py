"""
Reproducibility utilities for setting random seeds across different libraries.
"""

import random
import os
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_all_rng_states() -> dict:
    """Get current random number generator states for all libraries.
    
    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_all_rng_states(states: dict) -> None:
    """Set random number generator states for all libraries.
    
    Args:
        states: Dictionary containing RNG states from get_all_rng_states()
    """
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    
    if states['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])
