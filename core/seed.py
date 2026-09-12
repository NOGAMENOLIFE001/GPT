"""
Seed Management

Ensures reproducibility across runs.
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed() -> int:
    """Get current seed from torch."""
    return torch.initial_seed() & 0xFFFFFFFF


if __name__ == "__main__":
    set_seed(42)
    print(f"Random: {random.random()}")
    print(f"Numpy: {np.random.random()}")
    print(f"Torch: {torch.rand(1).item()}")