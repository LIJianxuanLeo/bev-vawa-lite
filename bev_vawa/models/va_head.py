"""VA branch: given latent state z_t, score each of the K fixed anchor waypoints
and optionally refine them by a small (dx, dy) offset."""
from __future__ import annotations
import torch
from torch import nn


class VAHead(nn.Module):
    def __init__(self, latent_dim: int = 128, n_candidates: int = 5, refine: bool = True,
                 hidden: int = 128):
        super().__init__()
        self.K = n_candidates
        self.refine = refine
        self.score = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_candidates),
        )
        if refine:
            self.offset = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n_candidates * 2),
            )

    def forward(self, z: torch.Tensor) -> dict:
        logits = self.score(z)                         # (B, K)
        out = {"logits": logits}
        if self.refine:
            off = self.offset(z).view(z.shape[0], self.K, 2)
            out["offset"] = 0.3 * torch.tanh(off)      # bounded offset in meters
        return out
