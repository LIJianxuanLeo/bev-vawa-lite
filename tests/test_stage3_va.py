"""Stage 3 gate: BEV encoder + VA head train smoke test on a mini dataset."""
from __future__ import annotations
import torch
import pytest

from bev_vawa.utils import load_config, set_seed, get_device
from bev_vawa.data import generate_dataset
from bev_vawa.models import BEVVAWA
from bev_vawa.train import train_stage_a


def test_model_forward_and_size():
    cfg = load_config("configs/default.yaml")
    model = BEVVAWA(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 5_000_000, f"model too large: {n_params}"

    H, W = cfg["env"]["depth_wh"]
    depth = torch.zeros(2, 1, H, W)
    goal = torch.zeros(2, 2)
    out = model(depth, goal, use_wa=True)
    K = cfg["va"]["n_candidates"]
    assert out["va_logits"].shape == (2, K)
    assert out["waypoints"].shape == (2, K, 2)
    assert out["wa_risk_logit"].shape == (2, K)
    wp, k = model.select_waypoint(out)
    assert wp.shape == (2, 2)


def test_mps_forward_fast():
    import time
    cfg = load_config("configs/default.yaml")
    device = get_device()
    model = BEVVAWA(cfg).to(device)
    H, W = cfg["env"]["depth_wh"]
    x = torch.zeros(1, 1, H, W, device=device)
    g = torch.zeros(1, 2, device=device)
    # warm up
    for _ in range(3):
        _ = model(x, g, use_wa=True)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = model(x, g, use_wa=True)
    if device.type == "mps":
        torch.mps.synchronize()
    dt_ms = (time.time() - t0) / 5 * 1000
    assert dt_ms < 300, f"forward too slow: {dt_ms:.1f} ms"


def test_stage_a_smoke(tmp_path):
    set_seed(0)
    cfg = load_config("configs/default.yaml")
    # tiny model for speed
    cfg["train"]["batch_size"] = 4
    data_dir = tmp_path / "data"
    generate_dataset(cfg, str(data_dir), n_rooms=3, samples_per_room=6, seed=0, verbose=False)
    ckpt = tmp_path / "stage_a.pt"
    res = train_stage_a(cfg, str(data_dir), str(ckpt), epochs=1, max_batches=3)
    assert ckpt.exists()
    assert res["loss"] < 1e4  # just sanity; real convergence checked elsewhere
