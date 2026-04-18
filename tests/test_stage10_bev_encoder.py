"""Stage 10 gate: geometric BEV encoder.

Covers:
  * ``GeometryLift`` shape, value-range, channel-mask correctness.
  * ``BEVEncoder`` forward / forward_seq shapes.
  * No-NaN / CPU-only smoke test.

Runs on every host (pure PyTorch, no habitat-sim needed).
"""
from __future__ import annotations
import torch

from bev_vawa.models.geometry_lift import GeometryLift
from bev_vawa.models.bev_encoder import BEVEncoder


# ---------------------------------------------------------------- geometry lift
def _synth_wall_depth(B: int = 2, H: int = 64, W: int = 64, d: float = 2.0) -> torch.Tensor:
    """A depth image of a flat wall at distance ``d`` metres."""
    return torch.full((B, 1, H, W), float(d))


def test_geometry_lift_shape_and_channels():
    lift = GeometryLift(grid_size=64, channels_enabled=(1, 1, 1))
    depth = _synth_wall_depth()
    goal = torch.tensor([[1.5, 0.0], [1.5, 0.3]])
    bev = lift(depth, goal)
    assert bev.shape == (2, 3, 64, 64)
    assert torch.isfinite(bev).all()
    assert bev[:, 0].min() >= 0.0 and bev[:, 0].max() <= 1.0
    assert bev[:, 1].min() >= 0.0 and bev[:, 1].max() <= 1.0
    assert bev[:, 2].min() >= 0.0 and bev[:, 2].max() <= 1.0 + 1e-5


def test_geometry_lift_occupancy_places_points_in_range():
    lift = GeometryLift(grid_size=64, bev_range=(0.0, 3.0, -1.5, 1.5),
                        channels_enabled=(1, 0, 0))
    depth = _synth_wall_depth(d=2.0)
    bev = lift(depth, None)
    occ = bev[:, 0]
    row_expected = int(2.0 / 3.0 * 64)
    occupied_rows = (occ.sum(dim=-1) > 0).nonzero()[:, 1]
    assert occupied_rows.numel() > 0
    assert abs(int(occupied_rows.float().mean()) - row_expected) <= 3


def test_geometry_lift_free_space_fills_near_robot():
    lift = GeometryLift(grid_size=64, channels_enabled=(1, 1, 0))
    depth = _synth_wall_depth(d=2.0)
    bev = lift(depth, None)
    free = bev[0, 1]
    occ = bev[0, 0]
    first_occ_row = (occ.sum(dim=-1) > 0).nonzero()
    assert first_occ_row.numel() > 0
    r0 = int(first_occ_row[0, 0])
    assert free[:r0].sum() > 0
    assert free[r0 + 5:].sum() == 0


def test_geometry_lift_channels_disabled_are_zero():
    lift = GeometryLift(grid_size=64, channels_enabled=(1, 0, 0))
    depth = _synth_wall_depth()
    bev = lift(depth, None)
    assert bev[:, 1].abs().max() == 0.0
    assert bev[:, 2].abs().max() == 0.0


def test_geometry_lift_goal_prior_peaks_near_bearing():
    lift = GeometryLift(grid_size=64, channels_enabled=(0, 0, 1),
                        goal_sector_sigma_rad=0.3)
    goal = torch.tensor([[1.0, 0.0]])
    bev = lift(torch.zeros(1, 1, 64, 64), goal)
    heat = bev[0, 2]
    center_col = 32
    assert heat[:, center_col].mean() > heat[:, 0].mean()
    assert heat[:, center_col].mean() > heat[:, -1].mean()


# ---------------------------------------------------------------- encoder
def test_bev_encoder_forward_shape():
    enc = BEVEncoder(grid_size=64, latent_dim=128, cnn_channels=(16, 32, 48))
    depth = _synth_wall_depth(B=3, H=128, W=128, d=1.5)
    goal = torch.zeros(3, 2)
    z = enc(depth, goal)
    assert z.shape == (3, 128)
    assert torch.isfinite(z).all()


def test_bev_encoder_forward_seq_shapes():
    enc = BEVEncoder(grid_size=32, latent_dim=64, cnn_channels=(8, 16, 24))
    T = 4
    depth_seq = torch.rand(2, T, 1, 64, 64) * 2.0
    goal_seq = torch.zeros(2, T, 2)
    out, h = enc.forward_seq(depth_seq, goal_seq)
    assert out.shape == (2, T, 64)
    assert torch.isfinite(out).all()
    assert isinstance(h, tuple) and h[0].shape == (2, 64)


def test_bev_encoder_exposes_stage_c_submodules():
    """Stage-C fine-tuning unfreezes bev_pool / fc_pool / input_proj by name;
    the encoder must expose them so the shared trainer code works unchanged."""
    enc = BEVEncoder()
    for name in ("bev_pool", "fc_pool", "input_proj"):
        assert hasattr(enc, name), f"encoder missing stage-C submodule {name!r}"


def test_bev_encoder_gradients_flow():
    enc = BEVEncoder(grid_size=32, latent_dim=32, cnn_channels=(8, 16, 24))
    depth = _synth_wall_depth(B=2, H=64, W=64, d=1.5)
    goal = torch.zeros(2, 2)
    z = enc(depth, goal)
    loss = z.pow(2).sum()
    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert max(grad_norms) > 0.0
