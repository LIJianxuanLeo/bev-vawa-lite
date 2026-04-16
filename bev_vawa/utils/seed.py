"""Global seeding for numpy / torch / python random."""
from __future__ import annotations
import os
import random
import numpy as np


def set_all(seed: int) -> None:
    """Seed every stochastic backend we use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:  # torch optional for env-only use
        pass
