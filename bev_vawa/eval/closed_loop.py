"""Closed-loop evaluation in the procedural MuJoCo env.

A *policy* is any callable with signature::

    policy(obs: dict, env_cfg: dict) -> (v: float, omega: float)

where ``obs`` follows ``NavEnv``'s observation spec. We provide several
built-in policies in ``bev_vawa/eval/policies.py``; here we only supply the
execution loop and metric aggregation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import time
import numpy as np

from ..envs import NavEnv
from ..envs.occupancy import rasterize, astar_path, world_to_cell, cell_to_world
from ..envs.pib_generator import RoomSpec, sample_room
from .metrics import summarize, spl_score
from ..utils import get_logger

log = get_logger(__name__)


Policy = Callable[[dict, dict], tuple[float, float]]


def _shortest_path_len(room: RoomSpec, cell_m: float) -> Optional[float]:
    grid = rasterize(room, cell_m=cell_m)
    s = world_to_cell(*room.start, room, cell_m)
    g = world_to_cell(*room.goal, room, cell_m)
    path = astar_path(grid, s, g)
    if path is None:
        return None
    pts = np.array([cell_to_world(r, c, room, cell_m) for (r, c) in path])
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(segs.sum())


def run_episode(env: NavEnv, policy: Policy, cfg: dict) -> dict:
    """Run one episode and return per-episode stats."""
    env_cfg = cfg["env"]
    obs = env._get_obs()
    agent_path = 0.0
    prev_xy = np.asarray(obs["pose"][:2]).copy()
    latencies = []
    steps = 0
    success = False
    while True:
        t0 = time.time()
        v, w = policy(obs, cfg)
        latencies.append((time.time() - t0) * 1000.0)
        step = env.step((v, w))
        obs = step.obs
        xy = np.asarray(obs["pose"][:2])
        agent_path += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy.copy()
        steps += 1
        if step.info["reached"]:
            success = True
            break
        if step.done:
            break
    shortest = _shortest_path_len(env.room, env_cfg["occupancy_cell_m"]) or float("inf")
    return {
        "success": bool(success),
        "shortest": float(shortest),
        "agent": float(agent_path),
        "collisions": int(env.n_collisions),
        "steps": int(steps),
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def run_eval(cfg: dict, policy: Policy, n_episodes: int, seed: int = 12345,
             episode_seeds: Optional[list[int]] = None) -> dict:
    """Run ``n_episodes`` with different room seeds; return aggregate metrics."""
    rng = np.random.default_rng(seed)
    if episode_seeds is None:
        episode_seeds = [int(rng.integers(1 << 30)) for _ in range(n_episodes)]
    env = NavEnv(cfg["env"], seed=seed)
    eps = []
    for i, s in enumerate(episode_seeds):
        env.reset(seed=s)
        eps.append(run_episode(env, policy, cfg))
    env.close()
    summary = summarize(eps)
    summary["per_episode"] = eps
    return summary
