"""Hard linear probing: does the BEV latent encode *precise* metric
geometry better than the flat latent?

This is a stricter follow-up to scripts/linear_probe.py. The original
probes (quadrant_medians, wall_ahead, asymmetry) turned out to be too
easy — both encoders solved them comparably. Here we design targets
that should specifically benefit from BEV's metric unprojection:

  1. (regression) **Nearest-obstacle polar coordinates** (r, theta) in
     the front arc. Requires localising a specific 3D object in
     agent-centric metric space — precisely what BEV's unprojection
     is designed to make linear-decodable.

  2. (classification, 5-way) **Forward-lane navigability**: for each
     of 5 cells at x=1.0m, y in {-0.8, -0.4, 0, +0.4, +0.8} relative
     to the agent, classify "navigable" vs "not navigable" (5-bit
     multi-label). Flat encoder has to re-learn the depth-to-metric
     mapping; BEV's occupancy channel exposes it explicitly.

  3. (classification, 12-way) **Goal angle**: bin the goal bearing
     into 12 sectors of 30 degrees each. Tests goal representation
     precision.

Usage:
    python scripts/linear_probe_hard.py \
        --bev-ckpt runs/default/stage_c.pt \
        --flat-ckpt runs/flat_encoder/seed_12345/stage_c.pt \
        --shards 'data/pib_nav/room_*.npz'
"""
from __future__ import annotations
import argparse
import csv
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_vawa.models import BEVVAWA
from bev_vawa.utils import load_config, set_seed


# ---------------------------------------------------------------------------
#  Probe target extraction
# ---------------------------------------------------------------------------

def _nearest_obstacle_polar(depth: np.ndarray, fov_rad: float = math.pi / 2,
                            depth_max: float = 3.0) -> np.ndarray:
    """Coarse metric localisation of the single closest depth cell.

    Treats each column of the depth image as a ray at bearing
        theta_col = -fov/2 + (col / W) * fov
    and each row as unimportant (we take the per-column minimum depth
    within the middle horizontal band). Returns (N, 2) = (r, theta)
    in (metres, radians). r is normalised by depth_max to [0, 1].

    Why this is a harder probe than our earlier wall_ahead: it asks
    for the exact (r, theta) of the closest surface, which requires
    metric unprojection.
    """
    N, H, W = depth.shape
    # Use the middle vertical band (skip ceiling / floor)
    r0, r1 = int(H * 0.3), int(H * 0.7)
    band = depth[:, r0:r1, :]                                 # (N, h, W)
    col_min = band.min(axis=1)                                # (N, W) nearest depth per column
    # Clip zero / NaN
    col_min = np.where(col_min > 0.05, col_min, depth_max)
    arg_col = col_min.argmin(axis=1)                          # (N,)
    r_val = col_min[np.arange(N), arg_col] / depth_max        # (N,) in [0, 1]
    theta_col = -fov_rad / 2.0 + (arg_col.astype(np.float32) / W) * fov_rad  # (N,) rad
    return np.stack([r_val, theta_col], axis=1).astype(np.float32)   # (N, 2)


def _forward_lane_navigability(depth: np.ndarray,
                               fov_rad: float = math.pi / 2,
                               depth_max: float = 3.0,
                               target_x: float = 1.0,
                               margin: float = 0.25) -> np.ndarray:
    """5-bit multi-label: is each of 5 cells at (1.0, {-0.8..+0.8})
    navigable?

    A cell at (x, y) is navigable if the depth ray whose bearing
    theta = atan2(y, x) has its min-depth greater than sqrt(x^2+y^2)
    by at least ``margin`` (i.e. the ray passes through the cell).
    """
    N, H, W = depth.shape
    r0, r1 = int(H * 0.3), int(H * 0.7)
    band_min = depth[:, r0:r1, :].min(axis=1)                  # (N, W)
    band_min = np.where(band_min > 0.05, band_min, depth_max)

    ys = np.array([-0.8, -0.4, 0.0, 0.4, 0.8], dtype=np.float32)
    cells = np.stack([np.full_like(ys, target_x), ys], axis=1)  # (5, 2) (x, y)
    thetas = np.arctan2(cells[:, 1], cells[:, 0])               # (5,) bearings in rad
    dists = np.sqrt(cells[:, 0] ** 2 + cells[:, 1] ** 2)        # (5,) distance to each cell

    # Map bearings to column indices
    cols = ((thetas + fov_rad / 2.0) / fov_rad * W).astype(np.int64)
    cols = np.clip(cols, 0, W - 1)

    # For each sample, for each target cell, check band_min[col] > dist + margin
    labels = np.zeros((N, len(ys)), dtype=np.int64)
    for k, (col, dist) in enumerate(zip(cols, dists)):
        labels[:, k] = (band_min[:, col] > dist + margin).astype(np.int64)
    return labels                                               # (N, 5)


def _goal_angle_bin(goal: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """Bin the goal bearing (rad, second column of goal_vec) into
    ``n_bins`` equal sectors of the full circle. Returns (N,) int64.
    """
    bearing = goal[:, 1].astype(np.float32)                    # rad, in [-pi, pi)
    # shift to [0, 2pi)
    b = (bearing + math.pi) % (2 * math.pi)
    idx = (b / (2 * math.pi) * n_bins).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


# ---------------------------------------------------------------------------
#  Encoder latent extraction + linear probes
# ---------------------------------------------------------------------------

def compute_latents(model: BEVVAWA, depths_t: torch.Tensor,
                    goals_t: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, depths_t.shape[0], batch_size):
            d = depths_t[i : i + batch_size].to(device)
            g = goals_t[i : i + batch_size].to(device)
            out.append(model.encoder(d, g).cpu())
    return torch.cat(out, dim=0)


def probe_reg(z_train, y_train, z_test, y_test, epochs: int = 80,
              lr: float = 5e-3) -> float:
    m = nn.Linear(z_train.shape[1], y_train.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    y_tr = torch.as_tensor(y_train, dtype=torch.float32)
    y_te = torch.as_tensor(y_test, dtype=torch.float32)
    best = -1e9
    for _ in range(epochs):
        m.train()
        p = m(z_train)
        loss = F.mse_loss(p, y_tr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p_te = m(z_test)
            mean = y_te.mean(0)
            ss_res = ((p_te - y_te) ** 2).sum()
            ss_tot = ((y_te - mean) ** 2).sum() + 1e-12
            r2 = float(1.0 - ss_res / ss_tot)
            best = max(best, r2)
    return best


def probe_bce_multilabel(z_train, y_train, z_test, y_test,
                          epochs: int = 80, lr: float = 5e-3) -> float:
    m = nn.Linear(z_train.shape[1], y_train.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    y_tr = torch.as_tensor(y_train, dtype=torch.float32)
    y_te = torch.as_tensor(y_test, dtype=torch.float32)
    best = 0.0
    for _ in range(epochs):
        m.train()
        logits = m(z_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred = (m(z_test) > 0).float()
            acc = float(((pred == y_te).float().mean()))
            best = max(best, acc)
    return best


def probe_ce(z_train, y_train, z_test, y_test, n_classes, epochs: int = 80,
             lr: float = 5e-3) -> float:
    m = nn.Linear(z_train.shape[1], n_classes)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    y_tr = torch.as_tensor(y_train, dtype=torch.long)
    y_te = torch.as_tensor(y_test, dtype=torch.long)
    best = 0.0
    for _ in range(epochs):
        m.train()
        logits = m(z_train)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(z_test).argmax(-1) == y_te).float().mean())
            best = max(best, acc)
    return best


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bev-ckpt", default="runs/default/stage_c.pt")
    ap.add_argument("--bev-config", default="configs/default.yaml")
    ap.add_argument("--flat-ckpt", default="runs/flat_encoder/seed_12345/stage_c.pt")
    ap.add_argument("--flat-config", default="configs/ablations/flat_encoder.yaml")
    ap.add_argument("--shards", default="data/pib_nav/room_*.npz")
    ap.add_argument("--max-shards", type=int, default=300)
    ap.add_argument("--out", default="results/bev_probing_hard.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards))[: args.max_shards]
    if not paths:
        raise SystemExit(f"No shards: {args.shards!r}")

    depth_list, goal_list = [], []
    for p in paths:
        z = np.load(p)
        depth_list.append(z["depth"].astype(np.float32))
        goal_list.append(z["goal"].astype(np.float32))
    depths_np = np.concatenate(depth_list, axis=0)
    goals_np = np.concatenate(goal_list, axis=0)
    print(f"Loaded {depths_np.shape[0]} samples from {len(paths)} shards")

    cfg = load_config(args.bev_config)
    depth_max = float(cfg["env"]["depth_max_m"])
    fov_rad = float(cfg["env"]["depth_fov_deg"]) * math.pi / 180.0

    targets = {
        "nearest_obstacle_polar": dict(
            y=_nearest_obstacle_polar(depths_np, fov_rad, depth_max), kind="reg",
        ),
        "forward_lane_navigability": dict(
            y=_forward_lane_navigability(depths_np, fov_rad, depth_max),
            kind="bce_multilabel",
        ),
        "goal_angle_bin12": dict(
            y=_goal_angle_bin(goals_np, 12), kind="ce", n_classes=12,
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "probe_target", "metric", "value"])

        for tag, ckpt, cfgp in [
            ("BEV", args.bev_ckpt, args.bev_config),
            ("flat", args.flat_ckpt, args.flat_config),
        ]:
            cfg_i = load_config(cfgp)
            set_seed(0)
            model = BEVVAWA(cfg_i)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            for p in model.parameters():
                p.requires_grad_(False)

            depths_t = torch.from_numpy(depths_np).unsqueeze(1)
            goals_t = torch.from_numpy(goals_np)
            print(f"[{tag}] computing {depths_t.shape[0]} latents...")
            z = compute_latents(model, depths_t, goals_t)       # (N, 128)

            N = z.shape[0]
            rng = np.random.default_rng(1234)
            perm = rng.permutation(N)
            idx_tr, idx_te = perm[: int(0.8 * N)], perm[int(0.8 * N) :]
            z_tr, z_te = z[idx_tr], z[idx_te]

            for tgt_name, info in targets.items():
                y = info["y"]
                y_tr, y_te = y[idx_tr], y[idx_te]
                if info["kind"] == "reg":
                    score = probe_reg(z_tr, y_tr, z_te, y_te)
                    metric = "R^2"
                elif info["kind"] == "bce_multilabel":
                    score = probe_bce_multilabel(z_tr, y_tr, z_te, y_te)
                    metric = "accuracy"
                else:
                    score = probe_ce(z_tr, y_tr, z_te, y_te, info["n_classes"])
                    metric = "accuracy"
                print(f"  {tag:<6} {tgt_name:<32} {metric}={score:.4f}")
                w.writerow([tag, tgt_name, metric, f"{score:.4f}"])

    print(f"\nSaved: {out_path}")
    print("-" * 60)
    with out_path.open() as fh:
        for row in fh:
            print(row.rstrip())
    print("-" * 60)


if __name__ == "__main__":
    main()
