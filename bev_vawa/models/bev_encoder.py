"""Geometry-aware BEV encoder.

Performs pinhole depth-unprojection onto a local floor plane, producing a
multi-channel BEV (occupancy + visible free-space + goal-prior sector
heatmap) via ``GeometryLift``, then runs a small BEV CNN and an
(optional) LSTM-cell recurrent fusion.

Output API:
  * ``encode_single(depth, goal) -> (B, latent)``  — no recurrence
  * ``forward_seq(depth_seq, goal_seq, hidden=None) -> ((B, T, latent), hidden)``
  * ``forward(depth, goal) -> (B, latent)`` — convenience, calls ``encode_single``
"""
from __future__ import annotations
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .geometry_lift import GeometryLift


class BEVEncoder(nn.Module):
    def __init__(
        self,
        depth_wh: Sequence[int] = (128, 128),
        grid_size: int = 64,
        latent_dim: int = 128,
        cnn_channels: Sequence[int] = (32, 64, 96),
        recurrent: str = "lstm",
        fov_deg: float = 90.0,
        depth_max_m: float = 3.0,
        bev_range: Sequence[float] = (0.0, 3.0, -1.5, 1.5),
        channels_enabled: Sequence[int] = (1, 1, 1),
        goal_sector_sigma_rad: float = 0.35,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.lift = GeometryLift(
            grid_size=grid_size,
            bev_range=bev_range,
            fov_deg=fov_deg,
            depth_max_m=depth_max_m,
            channels_enabled=channels_enabled,
            goal_sector_sigma_rad=goal_sector_sigma_rad,
        )
        c1, c2, c3 = cnn_channels
        bev_channels = 3
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(bev_channels, c1, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),       # /2
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True),       # /4
        )
        self.bev_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc_pool = nn.Linear(c3 * 8 * 8, latent_dim)

        self.recurrent_type = recurrent
        goal_dim = 2
        self.input_proj = nn.Linear(latent_dim + goal_dim, latent_dim)
        if recurrent == "lstm":
            self.rnn = nn.LSTMCell(latent_dim, latent_dim)
        elif recurrent == "gru":
            self.rnn = nn.GRUCell(latent_dim, latent_dim)
        else:
            self.rnn = None

    def encode_single(self, depth: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        bev = self.lift(depth, goal)                       # (B, 3, G, G)
        f = self.bev_cnn(bev)
        z = self.bev_pool(f).flatten(1)
        z = self.fc_pool(z)
        fused = self.input_proj(torch.cat([z, goal], dim=-1))
        return F.relu(fused)

    def forward_seq(self, depth_seq: torch.Tensor, goal_seq: torch.Tensor,
                    hidden=None) -> tuple[torch.Tensor, tuple]:
        B, T = depth_seq.shape[:2]
        outs = []
        if self.rnn is None:
            for t in range(T):
                outs.append(self.encode_single(depth_seq[:, t], goal_seq[:, t]))
            return torch.stack(outs, dim=1), None
        if hidden is None:
            if isinstance(self.rnn, nn.LSTMCell):
                h = (depth_seq.new_zeros(B, self.latent_dim),
                     depth_seq.new_zeros(B, self.latent_dim))
            else:
                h = depth_seq.new_zeros(B, self.latent_dim)
        else:
            h = hidden
        for t in range(T):
            x = self.encode_single(depth_seq[:, t], goal_seq[:, t])
            if isinstance(self.rnn, nn.LSTMCell):
                h = self.rnn(x, h)
                outs.append(h[0])
            else:
                h = self.rnn(x, h)
                outs.append(h)
        return torch.stack(outs, dim=1), h

    def forward(self, depth: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.encode_single(depth, goal)
