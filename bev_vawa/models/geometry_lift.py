"""Geometric depth-to-BEV lift for the v2 encoder path.

Unlike the original ``BEVEncoder`` which approximates a top-down feature via
``AdaptiveAvgPool2d``, this module performs an **actual pinhole unprojection**
of each valid depth pixel into the robot's local floor plane, then rasterises
the resulting points into a (grid, grid) occupancy map.

Design principles (aligned with doc §3 and §4):

* **Lightweight.** Pinhole back-projection only — no 3-D frustum transformer,
  no learned depth distribution. Keeps the parameter budget close to the
  original encoder.
* **Deterministic & differentiable-safe.** Uses ``index_add_`` into a flat
  buffer. Gradients do not flow back through the lift (it's a discretisation),
  but they do flow through the small CNN that consumes the BEV, which is what
  we want.
* **MPS + CUDA + CPU compatible.** Pure tensor ops, no custom kernels; the
  scatter uses ``index_add_`` on a 1-D buffer to dodge older-MPS edge cases
  with multi-axis ``scatter``.
* **Three channels available from day one** (doc §4.2):

  1. ``occupancy`` — a cell is 1 if any depth ray terminates in it.
  2. ``free_space`` — cells *between* the camera and the first occupied cell
     along each depth ray are 1. Approximates visible free space.
  3. ``goal_prior`` — an angular sector (Gaussian in bearing) on the BEV
     whose mean is the goal bearing from ``goal_vec``.

  Step 1 of the plan uses only channel 0 via ``channels_enabled=[1,0,0]``;
  Step 3 flips the other two on. This file is written once for both so Step 3
  is a config-only change.

BEV frame convention (matches the rest of the repo):
  * x-forward (robot), y-left (robot)
  * grid row index ``ix`` maps x to [x_min, x_max]
  * grid col index ``iy`` maps y to [y_min, y_max]
"""
from __future__ import annotations
import math
from typing import Optional, Sequence

import torch
from torch import nn


def _camera_intrinsics(H: int, W: int, fov_deg: float) -> tuple[float, float, float, float]:
    """Pinhole model from horizontal FOV. Returns (fx, fy, cx, cy)."""
    fov_rad = math.radians(fov_deg)
    fx = 0.5 * W / math.tan(fov_rad / 2.0)
    fy = fx  # square pixels
    cx = 0.5 * (W - 1)
    cy = 0.5 * (H - 1)
    return fx, fy, cx, cy


class GeometryLift(nn.Module):
    """Depth image -> 3-channel local BEV occupancy / free / goal-prior map.

    Parameters
    ----------
    grid_size
        Output H = W of the BEV image. Default 64.
    bev_range
        (x_min, x_max, y_min, y_max) in metres, robot frame. Default keeps
        ``x ∈ [0, 3.0]`` forward, ``y ∈ [-1.5, 1.5]`` lateral (doc §3.3).
    fov_deg
        Horizontal field of view of the depth sensor. Must match the env.
    depth_max_m
        Valid depth clamp; pixels > this are treated as invalid.
    channels_enabled
        Length-3 bool/int list controlling which channels are computed.
        ``[1,0,0]`` → Layer 1 (occupancy only). ``[1,1,1]`` → Layer 2 (all).
        Disabled channels are zeroed in the output; the output tensor is
        always (B, 3, grid, grid) so downstream code does not branch.
    goal_sector_sigma_rad
        Stddev (radians) of the Gaussian that fills the goal sector heatmap.
    """

    def __init__(
        self,
        grid_size: int = 64,
        bev_range: Sequence[float] = (0.0, 3.0, -1.5, 1.5),
        fov_deg: float = 90.0,
        depth_max_m: float = 3.0,
        channels_enabled: Sequence[int] = (1, 1, 1),
        goal_sector_sigma_rad: float = 0.35,
    ) -> None:
        super().__init__()
        assert len(bev_range) == 4
        assert len(channels_enabled) == 3
        self.grid = int(grid_size)
        self.x_min = float(bev_range[0])
        self.x_max = float(bev_range[1])
        self.y_min = float(bev_range[2])
        self.y_max = float(bev_range[3])
        self.fov_deg = float(fov_deg)
        self.depth_max = float(depth_max_m)
        self.chan = tuple(bool(c) for c in channels_enabled)
        self.goal_sigma = float(goal_sector_sigma_rad)

    # ------------------------------------------------------------------ helpers
    def _pixel_to_robot(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """depth: (B, 1, H, W). Returns robot-frame (x_r, y_r, valid)."""
        B, _, H, W = depth.shape
        device = depth.device
        dtype = depth.dtype

        fx, fy, cx, cy = _camera_intrinsics(H, W, self.fov_deg)
        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
        v = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)

        d = depth
        # camera frame: x-right, y-down, z-forward
        Xc = (u - cx) * d / fx
        # Yc would give vertical offset; we flatten onto the floor plane so
        # the y-dim of the image is discarded here (no ego-height modelling;
        # doc §3.3 accepts this).
        Zc = d
        # robot frame: x-forward (= Zc), y-left (= -Xc)
        x_r = Zc
        y_r = -Xc
        valid = (d > 0.05) & (d < self.depth_max) & torch.isfinite(d)
        return x_r, y_r, valid

    def _robot_to_grid(self, x_r: torch.Tensor, y_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise robot-frame (x, y) to integer grid indices + in-bounds mask."""
        G = self.grid
        ix = ((x_r - self.x_min) / (self.x_max - self.x_min) * G).long()
        iy = ((y_r - self.y_min) / (self.y_max - self.y_min) * G).long()
        in_bounds = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
        return ix, iy, in_bounds

    # --------------------------------------------------------------- occupancy
    def _occupancy(self, ix: torch.Tensor, iy: torch.Tensor, mask: torch.Tensor,
                   B: int) -> torch.Tensor:
        """Scatter valid points into a (B, 1, G, G) binary occupancy grid."""
        G = self.grid
        device = ix.device
        flat_buf = torch.zeros(B * G * G, device=device, dtype=torch.float32)

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand_as(ix)
        flat = b_idx * (G * G) + ix * G + iy               # (B, 1, H, W)
        flat_masked = flat[mask]
        if flat_masked.numel() > 0:
            ones = torch.ones_like(flat_masked, dtype=torch.float32)
            flat_buf.index_add_(0, flat_masked, ones)
        occ = (flat_buf.view(B, 1, G, G) > 0).to(torch.float32)
        return occ

    # --------------------------------------------------------------- free-space
    def _free_space(self, occ: torch.Tensor) -> torch.Tensor:
        """For each grid column, mark cells from the robot (ix=0 side) up to
        the first occupied cell as free. Approximates visible free space along
        each column, since x=forward => columns with shared iy correspond to
        successive ranges along a (nearly) radial slice.

        Simpler and MPS-friendly than per-ray Bresenham in the image plane;
        captures the same qualitative signal the doc §4.2 asks for.
        """
        B, _, G, _ = occ.shape
        # cumulative "any occupied so far" along the x (row) axis
        cum = occ.cumsum(dim=2) > 0                        # (B, 1, G, G)
        # Only columns where *some* cell was eventually observed can carry
        # free-space evidence; columns with zero depth returns are unknown,
        # not free.
        col_has_obs = (occ.sum(dim=2, keepdim=True) > 0)   # (B, 1, 1, G)
        # free = column has evidence AND not yet seen an obstacle AND current
        # cell is not itself occupied
        free = (~cum) & (occ < 0.5) & col_has_obs
        return free.to(torch.float32)

    # --------------------------------------------------------------- goal prior
    def _goal_prior(self, goal: torch.Tensor) -> torch.Tensor:
        """goal: (B, 2) with (distance, bearing). Returns (B, 1, G, G) heatmap.

        Implements doc §4.2 method A: a Gaussian sector in bearing centred on
        ``goal[:, 1]``. The radial profile is uniform over the covered range.
        """
        B = goal.shape[0]
        device = goal.device
        dtype = goal.dtype
        G = self.grid

        # grid cell centres in robot frame
        xs = torch.linspace(self.x_min, self.x_max, G, device=device, dtype=dtype)
        ys = torch.linspace(self.y_min, self.y_max, G, device=device, dtype=dtype)
        # (G, G): row=ix, col=iy
        X = xs.view(G, 1).expand(G, G)
        Y = ys.view(1, G).expand(G, G)
        bearing_grid = torch.atan2(Y, X.clamp(min=1e-6))    # (G, G) in [-pi/2, pi/2]

        # goal bearing per batch
        bearing_goal = goal[:, 1].view(B, 1, 1)             # (B, 1, 1)

        # wrap-safe angular difference
        d = bearing_grid.unsqueeze(0) - bearing_goal        # (B, G, G)
        d = (d + math.pi) % (2 * math.pi) - math.pi

        heat = torch.exp(-(d ** 2) / (2 * self.goal_sigma ** 2))  # (B, G, G)
        # mask to only x-forward cells (already true by construction of xs >= 0)
        return heat.unsqueeze(1).to(dtype)                   # (B, 1, G, G)

    # ----------------------------------------------------------------- forward
    def forward(self, depth: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        depth
            (B, 1, H, W) in metres.
        goal
            Optional (B, 2) = (distance, bearing). Only required when the goal
            channel is enabled.

        Returns
        -------
        bev
            (B, 3, grid, grid). Disabled channels are zeroed.
        """
        assert depth.dim() == 4 and depth.shape[1] == 1, \
            f"expected (B, 1, H, W), got {tuple(depth.shape)}"
        B = depth.shape[0]
        G = self.grid
        out = depth.new_zeros(B, 3, G, G)

        x_r, y_r, valid = self._pixel_to_robot(depth)
        ix, iy, in_bounds = self._robot_to_grid(x_r, y_r)
        mask = valid & in_bounds

        if self.chan[0] or self.chan[1]:
            occ = self._occupancy(ix, iy, mask, B)          # (B, 1, G, G)
        else:
            occ = depth.new_zeros(B, 1, G, G)

        if self.chan[0]:
            out[:, 0:1] = occ
        if self.chan[1]:
            out[:, 1:2] = self._free_space(occ)
        if self.chan[2]:
            if goal is None:
                raise ValueError("goal_prior channel enabled but goal is None")
            out[:, 2:3] = self._goal_prior(goal)

        return out
