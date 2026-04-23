"""Visualise BEV encoder's 3 channels on archetypal PIB-Nav observations.

Qualitative support for the "geometric BEV is the right shared
representation" claim (§1 Contribution 1). Picks 4 depth observations
from the PIB-Nav shards that span the interesting scene regime
(open / wall-facing / cluttered / corridor), runs the GeometryLift
on each, and renders a 4x5 grid:

    row  depth (128x128) | occupancy | free-space | goal-prior | flat-encoder input (pooled)
    --- OPEN             |           |           |            |
    --- WALL AHEAD       |           |           |            |
    --- CLUTTERED        |           |           |            |
    --- CORRIDOR         |           |           |            |

The last column ("flat-encoder input") is
``F.adaptive_avg_pool2d(depth, (64, 64))`` — the exact input the flat
ablation variant sees. Contrasting the BEV channels with the plain
pooled depth gives a visual argument for why the geometric lift
provides more useful structure.

Saves: ``results/bev_channels.png`` and ``results/bev_channels.pdf``.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

from bev_vawa.models.geometry_lift import GeometryLift
from bev_vawa.utils import load_config


def _pick_archetypal_indices(depths: np.ndarray) -> dict[str, int]:
    """Find 4 'interesting' depth maps from a shard by simple heuristics.

    - OPEN: depth quantile 0.95 is the largest (lots of open space)
    - WALL_AHEAD: mean forward-arc depth < 1.0 m (front blocked)
    - CLUTTERED: std of depth high (many short+long-range mix)
    - CORRIDOR: big left/right contrast (walls on sides, free ahead)
    """
    N, H, W = depths.shape
    r0, r1 = int(H * 0.3), int(H * 0.7)
    c_mid0, c_mid1 = int(W * 0.4), int(W * 0.6)

    q95 = np.quantile(depths.reshape(N, -1), 0.95, axis=1)
    fwd_mean = depths[:, r0:r1, c_mid0:c_mid1].mean(axis=(1, 2))
    stds = depths.std(axis=(1, 2))

    # corridor: side walls close, front open
    left_mean = depths[:, r0:r1, : W // 4].mean(axis=(1, 2))
    right_mean = depths[:, r0:r1, 3 * W // 4 :].mean(axis=(1, 2))
    front_mean = depths[:, r0:r1, W // 3 : 2 * W // 3].mean(axis=(1, 2))
    corridor_score = front_mean - 0.5 * (left_mean + right_mean)

    return {
        "OPEN": int(np.argmax(q95)),
        "WALL AHEAD": int(np.argmin(fwd_mean)),
        "CLUTTERED": int(np.argmax(stds)),
        "CORRIDOR": int(np.argmax(corridor_score)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shard", default="data/pib_nav/room_00000.npz",
                    help="which PIB-Nav shard to draw archetypes from")
    ap.add_argument("--out", default="results/bev_channels")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bev_cfg = cfg["bev"]
    env_cfg = cfg["env"]

    # Build the lift (same params as full_model.py uses)
    lift = GeometryLift(
        grid_size=bev_cfg["grid_size"],
        bev_range=tuple(bev_cfg.get("range", (0.0, 3.0, -1.5, 1.5))),
        fov_deg=float(env_cfg["depth_fov_deg"]),
        depth_max_m=float(env_cfg["depth_max_m"]),
        channels_enabled=tuple(bev_cfg.get("channels_enabled", (1, 1, 1))),
        goal_sector_sigma_rad=float(bev_cfg.get("goal_sector_sigma_rad", 0.35)),
    ).eval()

    # Load shard
    shard = np.load(args.shard)
    depths = shard["depth"].astype(np.float32)   # (N, H, W)
    goals = shard["goal"].astype(np.float32)     # (N, 2) = (dist, bearing)
    print(f"Loaded {depths.shape[0]} samples from {args.shard}")

    # Pick archetypes
    picks = _pick_archetypal_indices(depths)
    print("Archetypes:", picks)

    # Grid: rows = archetypes, cols = [depth, occ, free, goal_prior, pooled]
    archetype_names = list(picks.keys())
    n_rows = len(archetype_names)
    n_cols = 5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.3, n_rows * 2.3),
        squeeze=False,
    )

    col_titles = ["Depth (input)", "BEV: occupancy", "BEV: free-space",
                  "BEV: goal-prior", "Flat encoder input\n(pooled depth)"]

    # Channel cmaps: same for all 3 BEV channels
    bev_cmap = "viridis"
    depth_cmap = "gray_r"

    with torch.no_grad():
        for row, name in enumerate(archetype_names):
            i = picks[name]
            d_np = depths[i]
            g_np = goals[i]

            d = torch.from_numpy(d_np).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
            g = torch.from_numpy(g_np).unsqueeze(0)                # (1, 2)
            bev = lift(d, g).squeeze(0).cpu().numpy()              # (3, G, G)

            # Flat-encoder's input: AdaptiveAvgPool2d to (grid, grid)
            pooled = F.adaptive_avg_pool2d(
                d, (bev_cfg["grid_size"], bev_cfg["grid_size"])
            ).squeeze(0).squeeze(0).cpu().numpy()

            # Col 0: raw depth
            ax = axes[row][0]
            im = ax.imshow(d_np, cmap=depth_cmap, vmin=0, vmax=env_cfg["depth_max_m"])
            ax.set_title(f"{name}\n{col_titles[0]}" if row == 0 else name + "\n",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            # Col 1-3: BEV channels (flip so robot is at bottom, forward is up)
            for k, (ch_name, ch_idx) in enumerate([
                ("occupancy", 0), ("free-space", 1), ("goal-prior", 2)
            ]):
                ax = axes[row][k + 1]
                # Flip vertically for "forward is up"
                img = np.flipud(bev[ch_idx])
                ax.imshow(img, cmap=bev_cmap, vmin=0, vmax=1)
                if row == 0:
                    ax.set_title(col_titles[k + 1], fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                # draw agent (bottom-center) as a red dot
                G = bev.shape[-1]
                ax.plot(G // 2, G - 2, marker="^", color="red",
                        markersize=6, markeredgecolor="white")

            # Col 4: pooled depth (flat-encoder input)
            ax = axes[row][4]
            ax.imshow(pooled, cmap=depth_cmap,
                      vmin=0, vmax=env_cfg["depth_max_m"])
            if row == 0:
                ax.set_title(col_titles[4], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        "BEV channels vs flat-encoder input on 4 archetypal PIB-Nav depth "
        "observations\n(row = scene type; ▲ marks the agent in BEV views)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_base.with_suffix('.png')} + {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
