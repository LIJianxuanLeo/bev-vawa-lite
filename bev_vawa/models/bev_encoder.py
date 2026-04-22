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
        use_semantic: bool = False,
        semantic_classes: int = 16,
        semantic_feat_dim: int = 64,
        use_geometric_lift: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.use_semantic = bool(use_semantic)
        self.semantic_classes = int(semantic_classes)
        self.semantic_feat_dim = int(semantic_feat_dim)
        # Flag for the BEV ablation (see configs/ablations/flat_encoder.yaml).
        # When False, the geometric unprojection is skipped and the CNN
        # operates directly on a 1-channel pooled depth map resampled to
        # ``grid_size x grid_size``. This is the "flat" control variant
        # used to validate the claim that metric BEV is necessary —
        # parameter count differs by only ~1,600 (first Conv2d's in-channel
        # 3 vs 1), so any SR gap is attributable to the representation.
        self.use_geometric_lift = bool(use_geometric_lift)
        if self.use_geometric_lift:
            self.lift = GeometryLift(
                grid_size=grid_size,
                bev_range=bev_range,
                fov_deg=fov_deg,
                depth_max_m=depth_max_m,
                channels_enabled=channels_enabled,
                goal_sector_sigma_rad=goal_sector_sigma_rad,
            )
        else:
            self.lift = None
        c1, c2, c3 = cnn_channels
        # 3-channel (occ / free / goal-prior) when geometric lift is on;
        # 1-channel (raw pooled depth) for the flat ablation.
        in_channels = 3 if self.use_geometric_lift else 1
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),       # /2
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True),       # /4
        )
        self.bev_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc_pool = nn.Linear(c3 * 8 * 8, latent_dim)

        # Optional semantic branch: small parallel CNN over one-hot semantic
        # label map, pooled to a feat vector, concatenated to the depth
        # latent before goal fusion. Disabled when ``use_semantic=False``
        # so Stage-C checkpoints trained on depth-only shards keep working
        # bit-for-bit.
        if self.use_semantic:
            self.sem_cnn = nn.Sequential(
                nn.Conv2d(semantic_classes, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),  # /2
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),                # /4
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, semantic_feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.sem_cnn = None

        self.recurrent_type = recurrent
        goal_dim = 2
        sem_dim = self.semantic_feat_dim if self.use_semantic else 0
        self.input_proj = nn.Linear(latent_dim + goal_dim + sem_dim, latent_dim)
        if recurrent == "lstm":
            self.rnn = nn.LSTMCell(latent_dim, latent_dim)
        elif recurrent == "gru":
            self.rnn = nn.GRUCell(latent_dim, latent_dim)
        else:
            self.rnn = None

    # ------------------------------------------------------------------
    #  Semantic helper: accept either int label map (B, H, W) or pre-made
    #  one-hot (B, C, H, W). Returns (B, C, H, W) one-hot in floating dtype.
    # ------------------------------------------------------------------
    def _semantic_to_onehot(self, semantic: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if semantic.dim() == 3:
            # (B, H, W) int map → one-hot (B, C, H, W)
            semantic = semantic.long().clamp(0, self.semantic_classes - 1)
            oh = F.one_hot(semantic, num_classes=self.semantic_classes)
            oh = oh.permute(0, 3, 1, 2).contiguous().to(ref.dtype)
            return oh
        return semantic.to(ref.dtype)

    def encode_single(self, depth: torch.Tensor, goal: torch.Tensor,
                      semantic: "torch.Tensor | None" = None) -> torch.Tensor:
        if self.use_geometric_lift:
            bev = self.lift(depth, goal)                   # (B, 3, G, G)
        else:
            # Flat ablation: 1-channel pooled depth at the same grid size
            # as the BEV variant, so the downstream CNN / pool / FC stack
            # is identical and the only difference is the representation.
            # Goal is injected only through the final ``input_proj`` below,
            # not via the CNN spatial map — this is the deliberate
            # disadvantage of the flat variant that the BEV branch
            # receives via the goal_prior channel.
            bev = F.adaptive_avg_pool2d(depth, (self.grid_size, self.grid_size))
        f = self.bev_cnn(bev)
        z = self.bev_pool(f).flatten(1)
        z = self.fc_pool(z)
        if self.use_semantic:
            if semantic is None:
                # Silent fallback: if the dataset stops providing semantic
                # (e.g. cross-eval on a non-HM3D scene), use zeros.
                B = depth.shape[0]
                sem_feat = depth.new_zeros(B, self.semantic_feat_dim)
            else:
                sem_oh = self._semantic_to_onehot(semantic, depth)
                sem_feat = self.sem_cnn(sem_oh)
            fused = self.input_proj(torch.cat([z, goal, sem_feat], dim=-1))
        else:
            fused = self.input_proj(torch.cat([z, goal], dim=-1))
        return F.relu(fused)

    def forward_seq(self, depth_seq: torch.Tensor, goal_seq: torch.Tensor,
                    hidden=None, semantic_seq=None) -> tuple[torch.Tensor, tuple]:
        B, T = depth_seq.shape[:2]
        outs = []
        if self.rnn is None:
            for t in range(T):
                sem_t = None if semantic_seq is None else semantic_seq[:, t]
                outs.append(self.encode_single(depth_seq[:, t], goal_seq[:, t], sem_t))
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
            sem_t = None if semantic_seq is None else semantic_seq[:, t]
            x = self.encode_single(depth_seq[:, t], goal_seq[:, t], sem_t)
            if isinstance(self.rnn, nn.LSTMCell):
                h = self.rnn(x, h)
                outs.append(h[0])
            else:
                h = self.rnn(x, h)
                outs.append(h)
        return torch.stack(outs, dim=1), h

    def forward(self, depth: torch.Tensor, goal: torch.Tensor,
                semantic: "torch.Tensor | None" = None) -> torch.Tensor:
        return self.encode_single(depth, goal, semantic)
