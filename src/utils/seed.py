"""Reproducibility utilities for setting random seeds across libraries.

Educational simulation system.
"""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch (if installed).

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
