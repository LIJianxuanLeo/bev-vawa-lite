"""Single source of truth for the compute device (MPS -> CPU fallback)."""
from __future__ import annotations
import os


def get_device(prefer: str | None = None):
    import torch
    if prefer:
        return torch.device(prefer)
    if os.environ.get("BEVVAWA_FORCE_CPU"):
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
