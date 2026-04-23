"""Linear probing: does the BEV encoder's latent z_t encode more
structured geometric information than the flat-encoder variant?

Protocol (Alain & Bengio 2016; He et al. SimCLR 2020):
    1. Freeze the encoder trained under config X.
    2. On a held-out set of PIB-Nav shards, compute z_t for every sample.
    3. Train a single linear layer z_t -> y on a set of probe targets
       that require metric geometric understanding but are NOT the
       training targets of the VA / WA heads.
    4. Evaluate probe accuracy / R^2 on a disjoint split.
    5. Compare BEV-encoder probe score vs flat-encoder probe score.

A higher probe score means the encoder latent is a better *representation*
for geometric queries, independent of the closed-loop SR the main model
achieves. This is a standard representation-learning evaluation
decoupled from policy performance.

Probe targets (derivable from the raw depth image, never seen by VA/WA):
    1. (regression) quadrant median depths, 4-dim
        - TL, TR, BL, BR median depth of depth[r0:r1, c0:c1]
    2. (classification) "wall directly ahead": bool(front arc median < 1.0)
    3. (classification) "obstacle asymmetry": which side is closer
        (left / right / balanced), 3-class

Output: results/bev_probing.csv with rows:
    encoder, target, metric, value, std

Usage:
    python scripts/linear_probe.py \
        --bev-ckpt runs/default/stage_c.pt \
        --flat-ckpt runs/flat_encoder/seed_12345/stage_c.pt \
        --shards data/pib_nav/room_*.npz \
        --out results/bev_probing.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_vawa.models import BEVVAWA
from bev_vawa.utils import load_config, set_seed


# ---------------------------------------------------------------------------
#  Probe-target extraction from raw depth
# ---------------------------------------------------------------------------

def _quadrant_medians(depth: np.ndarray, depth_max: float) -> np.ndarray:
    """Returns 4-d vector: median depth in top-left / top-right / bottom-left / bottom-right.

    Normalised to [0, 1] by dividing by depth_max."""
    N, H, W = depth.shape
    h2, w2 = H // 2, W // 2
    q = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        d = depth[i]
        q[i, 0] = np.median(d[:h2, :w2])   # TL
        q[i, 1] = np.median(d[:h2, w2:])   # TR
        q[i, 2] = np.median(d[h2:, :w2])   # BL
        q[i, 3] = np.median(d[h2:, w2:])   # BR
    return q / depth_max


def _wall_ahead(depth: np.ndarray) -> np.ndarray:
    """Binary: is the front-arc median depth < 1.0 m?

    Front arc = rows [0.3H, 0.7H], cols [0.4W, 0.6W].
    Returns (N,) int64.
    """
    N, H, W = depth.shape
    r0, r1 = int(H * 0.3), int(H * 0.7)
    c0, c1 = int(W * 0.4), int(W * 0.6)
    fwd_med = np.median(depth[:, r0:r1, c0:c1].reshape(N, -1), axis=1)
    return (fwd_med < 1.0).astype(np.int64)


def _asymmetry(depth: np.ndarray, tol_m: float = 0.3) -> np.ndarray:
    """3-class: which side is closer on the horizon arc?

    0 = balanced (|diff| < tol), 1 = left closer, 2 = right closer.
    Returns (N,) int64.
    """
    N, H, W = depth.shape
    r0, r1 = int(H * 0.3), int(H * 0.7)
    left_med = np.median(depth[:, r0:r1, : W // 3].reshape(N, -1), axis=1)
    right_med = np.median(depth[:, r0:r1, 2 * W // 3 :].reshape(N, -1), axis=1)
    diff = left_med - right_med
    out = np.zeros(N, dtype=np.int64)
    out[diff < -tol_m] = 1      # left closer (smaller depth on left)
    out[diff > +tol_m] = 2      # right closer
    return out


# ---------------------------------------------------------------------------
#  Encoder latent extraction
# ---------------------------------------------------------------------------

def compute_latents(model: BEVVAWA, depths_t: torch.Tensor,
                    goals_t: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Run frozen ``model.encoder(depth, goal)`` in chunks."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, depths_t.shape[0], batch_size):
            d = depths_t[i : i + batch_size].to(device)
            g = goals_t[i : i + batch_size].to(device)
            z = model.encoder(d, g)
            out.append(z.cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
#  Probing (linear-only, via sklearn-equivalent closed form)
# ---------------------------------------------------------------------------

def linear_probe_classification(z_train, y_train, z_test, y_test, n_classes,
                                epochs: int = 30, lr: float = 1e-2) -> float:
    """Train a linear classifier on frozen z, return test accuracy."""
    model = nn.Linear(z_train.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    y_test_t = torch.as_tensor(y_test, dtype=torch.long)
    for _ in range(epochs):
        model.train()
        logits = model(z_train)
        loss = F.cross_entropy(logits, y_train_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(z_test).argmax(dim=-1)
        return float((pred == y_test_t).float().mean())


def linear_probe_regression(z_train, y_train, z_test, y_test,
                            epochs: int = 30, lr: float = 1e-2) -> float:
    """Train a linear regressor on frozen z, return test R^2."""
    out_dim = y_train.shape[1]
    model = nn.Linear(z_train.shape[1], out_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    y_test_t = torch.as_tensor(y_test, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        pred = model(z_train)
        loss = F.mse_loss(pred, y_train_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(z_test)
        y_mean = y_test_t.mean(dim=0)
        ss_res = ((pred - y_test_t) ** 2).sum()
        ss_tot = ((y_test_t - y_mean) ** 2).sum() + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)
    return r2


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run_probe(tag: str, ckpt_path: str, config_path: str,
              depths_np, goals_np, targets: dict,
              writer, split_ratio: float = 0.8, seed: int = 0):
    """Load checkpoint under config, probe all targets, write CSV rows."""
    cfg = load_config(config_path)
    set_seed(seed)
    model = BEVVAWA(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    depths_t = torch.from_numpy(depths_np).unsqueeze(1)   # (N, 1, H, W)
    goals_t = torch.from_numpy(goals_np)                  # (N, 2)
    print(f"[{tag}] computing {depths_t.shape[0]} latents...")
    t0 = time.time()
    z = compute_latents(model, depths_t, goals_t)         # (N, 128)
    print(f"[{tag}] done in {time.time() - t0:.1f}s, z shape {tuple(z.shape)}")

    # Split once with a fixed seed so BEV and flat see the SAME train/test rows
    N = z.shape[0]
    rng = np.random.default_rng(1234)
    perm = rng.permutation(N)
    n_train = int(split_ratio * N)
    idx_train, idx_test = perm[:n_train], perm[n_train:]
    z_train, z_test = z[idx_train], z[idx_test]

    for tgt_name, tgt_info in targets.items():
        y = tgt_info["y"]
        kind = tgt_info["kind"]
        y_train, y_test = y[idx_train], y[idx_test]
        if kind == "regression":
            score = linear_probe_regression(z_train, y_train, z_test, y_test)
            metric = "R^2"
        else:
            score = linear_probe_classification(
                z_train, y_train, z_test, y_test, n_classes=tgt_info["n_classes"]
            )
            metric = "accuracy"
        print(f"  {tag:<10} {tgt_name:<20} {metric}={score:.4f}")
        writer.writerow([tag, tgt_name, metric, f"{score:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bev-ckpt", default="runs/default/stage_c.pt")
    ap.add_argument("--bev-config", default="configs/default.yaml")
    ap.add_argument("--flat-ckpt", default="runs/flat_encoder/seed_12345/stage_c.pt")
    ap.add_argument("--flat-config", default="configs/ablations/flat_encoder.yaml")
    ap.add_argument("--shards", default="data/pib_nav/room_0000*.npz",
                    help="glob pattern for shards to use")
    ap.add_argument("--max-shards", type=int, default=30,
                    help="cap how many shards to load (each has ~50 samples)")
    ap.add_argument("--out", default="results/bev_probing.csv")
    args = ap.parse_args()

    # Load pooled depth + goal arrays across multiple shards
    paths = sorted(glob.glob(args.shards))[: args.max_shards]
    if not paths:
        raise SystemExit(f"No shards matched {args.shards!r}")

    cfg = load_config(args.bev_config)
    depth_max = float(cfg["env"]["depth_max_m"])

    depth_list, goal_list = [], []
    for p in paths:
        z = np.load(p)
        depth_list.append(z["depth"].astype(np.float32))
        goal_list.append(z["goal"].astype(np.float32))
    depths_np = np.concatenate(depth_list, axis=0)   # (N, H, W)
    goals_np = np.concatenate(goal_list, axis=0)     # (N, 2)
    print(f"Loaded {depths_np.shape[0]} samples from {len(paths)} shards")

    # Build probe targets from raw depth (identical for both encoders)
    targets = {
        "quadrant_medians": {
            "y": _quadrant_medians(depths_np, depth_max),
            "kind": "regression",
        },
        "wall_ahead": {
            "y": _wall_ahead(depths_np),
            "kind": "classification",
            "n_classes": 2,
        },
        "asymmetry": {
            "y": _asymmetry(depths_np),
            "kind": "classification",
            "n_classes": 3,
        },
    }

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "probe_target", "metric", "value"])

        run_probe("BEV",  args.bev_ckpt,  args.bev_config,
                  depths_np, goals_np, targets, w)
        run_probe("flat", args.flat_ckpt, args.flat_config,
                  depths_np, goals_np, targets, w)

    print(f"\nSaved: {out_path}")
    print("Re-read to print final comparison table:")
    print("-" * 60)
    with out_path.open() as fh:
        for row in fh:
            print(row.rstrip())
    print("-" * 60)


if __name__ == "__main__":
    main()
