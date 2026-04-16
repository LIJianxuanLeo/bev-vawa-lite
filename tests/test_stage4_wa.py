"""Stage 4 gate: WA head + Stage-B/C smoke train + fusion sanity."""
from __future__ import annotations
import torch
import pytest

from bev_vawa.utils import load_config, set_seed
from bev_vawa.data import generate_dataset
from bev_vawa.models import BEVVAWA, fuse_scores
from bev_vawa.train import train_stage_a, train_stage_b, train_stage_c


def test_fusion_argmax():
    va = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    risk = torch.tensor([[-5.0, 5.0, -5.0, -5.0, -5.0]])   # anchor 1 is risky
    prog = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    unc = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0]])
    Q = fuse_scores(va, risk, prog, unc, alpha=1, beta=1, gamma=2, delta=0.5)
    assert int(Q.argmax(dim=-1).item()) != 1  # penalized away from risky anchor


def test_stage_b_smoke(tmp_path):
    set_seed(3)
    cfg = load_config("configs/default.yaml")
    cfg["train"]["batch_size"] = 4
    data_dir = tmp_path / "data"
    generate_dataset(cfg, str(data_dir), n_rooms=3, samples_per_room=6, seed=3, verbose=False)
    ck_a = tmp_path / "a.pt"
    ck_b = tmp_path / "b.pt"
    train_stage_a(cfg, str(data_dir), str(ck_a), epochs=1, max_batches=2)
    res = train_stage_b(cfg, str(data_dir), str(ck_a), str(ck_b), epochs=1, max_batches=2)
    assert ck_b.exists()
    assert res["loss"] < 1e4


def test_stage_c_smoke(tmp_path):
    set_seed(4)
    cfg = load_config("configs/default.yaml")
    cfg["train"]["batch_size"] = 4
    data_dir = tmp_path / "data"
    generate_dataset(cfg, str(data_dir), n_rooms=3, samples_per_room=6, seed=4, verbose=False)
    ck_a = tmp_path / "a.pt"
    ck_b = tmp_path / "b.pt"
    ck_c = tmp_path / "c.pt"
    train_stage_a(cfg, str(data_dir), str(ck_a), epochs=1, max_batches=2)
    train_stage_b(cfg, str(data_dir), str(ck_a), str(ck_b), epochs=1, max_batches=2)
    res = train_stage_c(cfg, str(data_dir), str(ck_b), str(ck_c), epochs=1, max_batches=2)
    assert ck_c.exists()
    assert res["loss"] < 1e4
