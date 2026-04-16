"""Shared BEV encoder: depth CNN -> lift-to-BEV -> LSTM -> latent z_t.

Design notes:
  * LSTM (not GRU) to dodge PyTorch-MPS GRU slowness.
  * The "lift-to-BEV" step is an approximate: we reshape the CNN feature map
    and pool it into a (grid, grid) 2-D grid before a 1x1 conv. This is a
    light surrogate for true depth-unprojection; it preserves the narrative
    ("shared BEV state") without needing camera intrinsics or a full frustum
    transformer, which would balloon the parameter count.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class BEVEncoder(nn.Module):
    def __init__(self, depth_wh=(128, 128), grid_size=64, latent_dim=128,
                 cnn_channels=(32, 64, 96), recurrent: str = "lstm"):
        super().__init__()
        c1, c2, c3 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, 5, stride=2, padding=2), nn.ReLU(inplace=True),   # /2
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # /4
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # /8
        )
        # after 3x stride-2, depth_wh -> depth_wh/8
        self.bev_grid = grid_size
        self.feat_c = c3
        # lift-to-BEV: 1x1 conv on the pooled feature after an adaptive pool to
        # (grid_size, grid_size)
        self.lift = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Conv2d(c3, latent_dim // 2, 1), nn.ReLU(inplace=True),
        )
        self.bev_pool = nn.Sequential(
            nn.Conv2d(latent_dim // 2, latent_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc_pool = nn.Linear(latent_dim * 8 * 8, latent_dim)

        # goal-conditioned recurrent update
        self.recurrent_type = recurrent
        goal_dim = 2
        self.input_proj = nn.Linear(latent_dim + goal_dim, latent_dim)
        if recurrent == "lstm":
            self.rnn = nn.LSTMCell(latent_dim, latent_dim)
        elif recurrent == "gru":
            self.rnn = nn.GRUCell(latent_dim, latent_dim)
        else:
            self.rnn = None  # no temporal fusion
        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    def encode_single(self, depth: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Encode a single (B, 1, H, W) depth frame + (B, 2) goal into a (B, latent) vector.
        No recurrence; use ``forward_seq`` for temporal input."""
        f = self.cnn(depth)
        bev = self.lift(f)
        z = self.bev_pool(bev).flatten(1)
        z = self.fc_pool(z)
        fused = self.input_proj(torch.cat([z, goal], dim=-1))
        return F.relu(fused)

    def forward_seq(self, depth_seq: torch.Tensor, goal_seq: torch.Tensor,
                     hidden=None) -> tuple[torch.Tensor, tuple]:
        """depth_seq: (B, T, 1, H, W), goal_seq: (B, T, 2). Returns (B, T, latent), hidden."""
        B, T = depth_seq.shape[:2]
        outs = []
        h = None
        if self.rnn is None:
            for t in range(T):
                outs.append(self.encode_single(depth_seq[:, t], goal_seq[:, t]))
            return torch.stack(outs, dim=1), None
        if hidden is None:
            if isinstance(self.rnn, nn.LSTMCell):
                h = (depth_seq.new_zeros(B, self.latent_dim), depth_seq.new_zeros(B, self.latent_dim))
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
        """Default: single-step encode (no recurrence)."""
        return self.encode_single(depth, goal)
