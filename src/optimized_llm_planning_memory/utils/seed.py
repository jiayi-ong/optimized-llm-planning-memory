"""
utils/seed.py
=============
Reproducible seeding across Python, NumPy, and PyTorch.

Usage
-----
    from optimized_llm_planning_memory.utils.seed import set_seed
    set_seed(42)

Call once at the start of a training run or evaluation script. For RL training,
also pass the seed to ``make_vec_env`` / ``CompressionEnv.reset(seed=N)`` so
that each worker's environment is independently reproducible.
"""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : Integer seed value. Must be in [0, 2**32 - 1].
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN ops (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
