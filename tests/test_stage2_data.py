"""Stage 2 gate: mini dataset generation + shapes + dataset round-trip."""
from __future__ import annotations
import numpy as np
import pytest

from bev_vawa.utils import load_config, set_seed
from bev_vawa.data import generate_dataset, NavShardDataset


def test_mini_generation(tmp_path):
    set_seed(0)
    cfg = load_config("configs/default.yaml")
    n = generate_dataset(cfg, str(tmp_path), n_rooms=3, samples_per_room=6, seed=0, verbose=False)
    assert n >= 2, f"only {n} shards written"

    ds = NavShardDataset(str(tmp_path), depth_max=cfg["env"]["depth_max_m"])
    assert len(ds) >= 8
    assert ds.anchors.shape == (cfg["va"]["n_candidates"], 2)

    sample = ds[0]
    K = cfg["va"]["n_candidates"]
    H, W = cfg["env"]["depth_wh"]
    assert sample["depth"].shape == (1, H, W)
    assert sample["goal"].shape == (2,)
    assert sample["expert_wp"].shape == (2,)
    assert sample["cand_collision"].shape == (K,)
    assert sample["cand_progress"].shape == (K,)
    assert 0 <= int(sample["best_k"]) < K
    assert sample["depth"].min().item() >= 0.0
    assert sample["depth"].max().item() <= 1.0 + 1e-5


def test_label_sanity(tmp_path):
    set_seed(1)
    cfg = load_config("configs/default.yaml")
    generate_dataset(cfg, str(tmp_path), n_rooms=5, samples_per_room=8, seed=1, verbose=False)
    ds = NavShardDataset(str(tmp_path), depth_max=cfg["env"]["depth_max_m"])
    # gather labels
    collisions = np.stack([ds[i]["cand_collision"].numpy() for i in range(len(ds))])
    progress = np.stack([ds[i]["cand_progress"].numpy() for i in range(len(ds))])
    assert collisions.shape[0] == len(ds)
    # not degenerate: not all 0 and not all 1
    p_coll = float(collisions.mean())
    assert 0.0 < p_coll < 1.0, f"collision rate is degenerate: {p_coll}"
    # progress should vary per anchor (positive and negative values)
    assert progress.min() < 0 < progress.max()


def test_dataset_roundtrip(tmp_path):
    set_seed(2)
    cfg = load_config("configs/default.yaml")
    generate_dataset(cfg, str(tmp_path), n_rooms=2, samples_per_room=4, seed=2, verbose=False)
    ds = NavShardDataset(str(tmp_path), depth_max=cfg["env"]["depth_max_m"])
    a = ds[0]
    b = ds[0]
    # deterministic under same key
    assert (a["depth"] == b["depth"]).all()
    assert (a["expert_wp"] == b["expert_wp"]).all()
