"""PointGoal navigation metrics: SR, SPL, collision rate, path-length ratio."""
from __future__ import annotations
from typing import List
import numpy as np


def spl_score(success: bool, shortest_path: float, agent_path: float) -> float:
    """Anderson et al. 2018 SPL. 0 if failed, else shortest / max(agent, shortest)."""
    if not success:
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
    plr = float(np.mean([e["agent"] / max(e["shortest"], 1e-6) for e in episodes if e["success"]]) if sr > 0 else float("nan"))
    lat = float(arr("latency_ms").mean())
    return {"n": n, "SR": sr, "SPL": spl, "CollisionRate": coll_rate, "PathLenRatio": plr, "LatencyMs": lat}
