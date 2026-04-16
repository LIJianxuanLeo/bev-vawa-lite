"""Offline dataset generator: teleport the robot along A*-smoothed expert paths,
render depth, compute per-anchor labels, dump to .npz shards.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from ..envs import NavEnv, sample_room
from ..envs.occupancy import rasterize, astar_path, world_to_cell, cell_to_world
from ..envs.pib_generator import RoomSpec
from .expert import (
    chaikin_smooth, path_resample, expert_waypoint_from_path,
    candidate_anchors, label_candidates, best_k_for_expert,
)
from ..utils import get_logger

log = get_logger(__name__)


def _astar_world(room: RoomSpec, cell_m: float) -> Optional[np.ndarray]:
    grid = rasterize(room, cell_m=cell_m)
    s = world_to_cell(*room.start, room, cell_m)
    g = world_to_cell(*room.goal, room, cell_m)
    path = astar_path(grid, s, g)
    if path is None:
        return None
    pts = np.array([cell_to_world(r, c, room, cell_m) for (r, c) in path], dtype=np.float32)
    return pts


def generate_one_room(
    env: NavEnv, rng: np.random.Generator, n_samples: int, cfg: dict
) -> Optional[dict]:
    """Generate ``n_samples`` training tuples along the expert path inside a single room."""
    env_cfg = cfg["env"]
    va_cfg = cfg["va"]
    cell_m = env_cfg["occupancy_cell_m"]

    # ensure room is solvable
    for _ in range(5):
        room = sample_room(rng, env_cfg)
        path_w = _astar_world(room, cell_m)
        if path_w is not None and len(path_w) >= 4:
            break
    else:
        return None
    env.reset(room=room, seed=int(rng.integers(1 << 30)))
    grid = rasterize(room, cell_m=cell_m)

    path_s = path_resample(chaikin_smooth(path_w, iterations=2), step_m=0.10)
    if len(path_s) < 4:
        return None

    K = int(va_cfg["n_candidates"])
    horizon = float(va_cfg["waypoint_horizon_m"])
    anchors = candidate_anchors(K, horizon)

    # sample pose indices along the path
    idx_pool = np.arange(len(path_s) - 1)
    if len(idx_pool) == 0:
        return None
    picks = rng.choice(idx_pool, size=min(n_samples, len(idx_pool)), replace=len(idx_pool) < n_samples)

    depths, goals, poses = [], [], []
    expert_wps, cand_coll, cand_prog, best_ks = [], [], [], []

    for i in picks:
        base = path_s[i]
        # orient robot tangent to path with noise
        nxt = path_s[min(i + 2, len(path_s) - 1)]
        tangent = nxt - base
        if np.linalg.norm(tangent) < 1e-6:
            continue
        yaw = float(np.arctan2(tangent[1], tangent[0]))
        # small position + yaw noise
        px = float(base[0] + rng.normal(0, 0.05))
        py = float(base[1] + rng.normal(0, 0.05))
        pyaw = float(yaw + rng.normal(0, 0.15))
        env.teleport(px, py, pyaw)
        obs = env._get_obs()
        pose = (px, py, pyaw)

        ewp = expert_waypoint_from_path(path_s, pose, horizon_m=horizon)
        coll, prog, _ = label_candidates(anchors, pose, room.goal, grid, room, cell_m)
        best_k = best_k_for_expert(anchors, ewp)

        depths.append(obs["depth"].astype(np.float32))
        goals.append(obs["goal_vec"].astype(np.float32))
        poses.append(obs["pose"].astype(np.float32))
        expert_wps.append(ewp.astype(np.float32))
        cand_coll.append(coll.astype(np.float32))
        cand_prog.append(prog.astype(np.float32))
        best_ks.append(np.int64(best_k))

    if not depths:
        return None
    return {
        "depth": np.stack(depths),                  # (N, H, W)
        "goal": np.stack(goals),                    # (N, 2)
        "pose": np.stack(poses),                    # (N, 3)
        "expert_wp": np.stack(expert_wps),          # (N, 2) robot frame
        "cand_collision": np.stack(cand_coll),      # (N, K) float in {0,1}
        "cand_progress": np.stack(cand_prog),       # (N, K)
        "best_k": np.stack(best_ks),                # (N,)
        "anchors": anchors.astype(np.float32),      # (K, 2)
    }


def generate_dataset(cfg: dict, out_dir: str, n_rooms: int, samples_per_room: int, seed: int = 0, verbose: bool = True):
    """Serial dataset generator. Writes one .npz per room. Fast enough on M4."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Build one env; re-use across rooms by calling reset(room=...).
    # We need a throwaway room for the first reset; NavEnv.__init__ does this internally.
    env = NavEnv(cfg["env"], seed=seed)

    written = 0
    for r in range(n_rooms):
        shard = generate_one_room(env, rng, samples_per_room, cfg)
        if shard is None:
            continue
        p = out / f"room_{r:05d}.npz"
        np.savez_compressed(p, **shard)
        written += 1
        if verbose and (r + 1) % 20 == 0:
            log.info(f"[{r+1}/{n_rooms}] rooms, {written} shards written to {out}")
    env.close()
    log.info(f"Done. {written} shards in {out}")
    return written
