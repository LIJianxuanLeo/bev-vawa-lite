"""PointGoal navigation metrics: SR, SPL, collision rate, path-length ratio."""
from __future__ import annotations
from typing import List
import numpy as np


def spl_score(success: bool, shortest_path: float, agent_path: float) -> float:
    """Anderson et al. 2018 SPL. 0 if failed, else shortest / max(agent, shortest).

    When the reference shortest path is unavailable (closed_loop falls back to
    ``inf`` if the A*-grid planner can't connect start→goal), we conservatively
    return 0 rather than propagating ``inf/inf = nan`` through the mean.
    """
    if not success:
        return 0.0
    if not np.isfinite(shortest_path) or shortest_path <= 0:
        return 0.0
    if not np.isfinite(agent_path) or agent_path <= 0:
        return 0.0
    return float(shortest_path / max(agent_path, shortest_path, 1e-6))


def summarize(episodes: List[dict]) -> dict:
    """episodes: list of dicts with keys success, shortest, agent, collisions, steps, latency_ms."""
    n = len(episodes)
    if n == 0:
        return {"n": 0}
    arr = lambda k: np.asarray([e[k] for e in episodes], dtype=np.float64)
    sr = float(arr("success").mean())
    spl = float(np.mean([spl_score(bool(e["success"]), e["shortest"], e["agent"]) for e in episodes]))
    collided = arr("collisions")
    coll_rate = float((collided > 0).mean())
    # PathLenRatio: mean over successful episodes with a valid reference path.
    plr_vals = [e["agent"] / e["shortest"] for e in episodes
                if e["success"] and np.isfinite(e["shortest"]) and e["shortest"] > 0]
    plr = float(np.mean(plr_vals)) if plr_vals else float("nan")
    lat = float(arr("latency_ms").mean())
    return {"n": n, "SR": sr, "SPL": spl, "CollisionRate": coll_rate, "PathLenRatio": plr, "LatencyMs": lat}
