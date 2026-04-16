"""Stage 6 gate: baselines + ablations smoke train."""
from __future__ import annotations
import torch
import pytest

from bev_vawa.utils import load_config, set_seed
from bev_vawa.data import generate_dataset
from bev_vawa.models import FPV_BC, BEV_VA, BEV_BC, BEVVAWA
from bev_vawa.train import train_baseline, train_stage_a, train_stage_b


def test_baseline_forward_shapes():
    cfg = load_config("configs/default.yaml")
    H, W = cfg["env"]["depth_wh"]
    K = cfg["va"]["n_candidates"]
    d = torch.zeros(2, 1, H, W)
    g = torch.zeros(2, 2)
    for cls in (FPV_BC, BEV_VA):
        m = cls(cfg)
        o = m(d, g)
        assert o["waypoints"].shape == (2, K, 2)
        wp, k = m.select_waypoint(o)
        assert wp.shape == (2, 2)
    m = BEV_BC(cfg)
    o = m(d, g)
    assert o["waypoints"].shape == (2, 1, 2)
    wp, k = m.select_waypoint(o)
    assert wp.shape == (2, 2)


def test_fpv_bc_smoke_train(tmp_path):
    set_seed(0)
    cfg = load_config("configs/default.yaml")
    cfg["train"]["batch_size"] = 4
    data_dir = tmp_path / "data"
    generate_dataset(cfg, str(data_dir), n_rooms=2, samples_per_room=6, seed=0, verbose=False)
    ckpt = tmp_path / "fpv.pt"
    res = train_baseline(FPV_BC, cfg, str(data_dir), str(ckpt), epochs=1, max_batches=2)
    assert ckpt.exists() and res["loss"] < 1e4


def test_bev_bc_smoke_train(tmp_path):
    set_seed(1)
    cfg = load_config("configs/default.yaml")
    cfg["train"]["batch_size"] = 4
    data_dir = tmp_path / "data"
    generate_dataset(cfg, str(data_dir), n_rooms=2, samples_per_room=6, seed=1, verbose=False)
    ckpt = tmp_path / "bev_bc.pt"
    res = train_baseline(BEV_BC, cfg, str(data_dir), str(ckpt), epochs=1, max_batches=2)
    assert ckpt.exists() and res["loss"] < 1e4


def test_ablation_configs_load():
    for name in ["no_wa", "no_unc", "h1", "k1", "k3"]:
        cfg = load_config(f"configs/ablations/{name}.yaml")
        assert "env" in cfg and "bev" in cfg and "va" in cfg and "wa" in cfg


def test_ablation_k1_model_forward():
    cfg = load_config("configs/ablations/k1.yaml")
    m = BEVVAWA(cfg)
    H, W = cfg["env"]["depth_wh"]
    d = torch.zeros(1, 1, H, W)
    g = torch.zeros(1, 2)
    out = m(d, g, use_wa=True)
    assert out["va_logits"].shape == (1, 1)
    assert out["waypoints"].shape == (1, 1, 2)


def test_ablation_h1_model_forward():
    cfg = load_config("configs/ablations/h1.yaml")
    m = BEVVAWA(cfg)
    H, W = cfg["env"]["depth_wh"]
    d = torch.zeros(1, 1, H, W)
    g = torch.zeros(1, 2)
    out = m(d, g, use_wa=True)
    assert out["wa_risk_logit"].shape == (1, cfg["va"]["n_candidates"])
