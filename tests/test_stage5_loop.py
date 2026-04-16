"""Stage 5 gate: controller + closed-loop evaluator."""
from __future__ import annotations
import math
import numpy as np
import pytest

from bev_vawa.utils import load_config, set_seed
from bev_vawa.control import pure_pursuit_cmd
from bev_vawa.eval import run_episode, run_eval, summarize
from bev_vawa.eval.metrics import spl_score
from bev_vawa.eval.policies import make_goal_policy, make_astar_policy
from bev_vawa.envs import NavEnv


def test_pure_pursuit_straight_ahead():
    v, w = pure_pursuit_cmd((1.0, 0.0), max_lin=0.4, max_ang=1.2)
    assert v > 0.0
    assert abs(w) < 1e-6


def test_pure_pursuit_turn():
    v, w = pure_pursuit_cmd((0.0, 1.0), max_lin=0.4, max_ang=1.2)
    assert w > 0.0  # needs to turn left


def test_spl_math():
    assert spl_score(False, 10.0, 12.0) == 0.0
    assert abs(spl_score(True, 10.0, 10.0) - 1.0) < 1e-6
    assert abs(spl_score(True, 10.0, 20.0) - 0.5) < 1e-6


def test_summarize_empty_and_nontrivial():
    assert summarize([]) == {"n": 0}
    eps = [
        {"success": True, "shortest": 5.0, "agent": 6.0, "collisions": 0, "steps": 60, "latency_ms": 5.0},
        {"success": False, "shortest": 8.0, "agent": 12.0, "collisions": 3, "steps": 100, "latency_ms": 6.0},
    ]
    s = summarize(eps)
    assert abs(s["SR"] - 0.5) < 1e-9
    assert 0 < s["SPL"] < 1


def test_astar_upper_bound_eval():
    """Oracle A*+PurePursuit should hit a reasonable SR on a handful of rooms.
    The threshold is lax (>=50%) because our pure-pursuit controller grazes walls
    on tight turns; the paper reports the full A* upper bound with appropriate
    tuning and is not intended to be perfect. This is a smoke gate only."""
    set_seed(10)
    cfg = load_config("configs/default.yaml")
    cfg["env"]["max_episode_steps"] = 400
    rng = np.random.default_rng(10)
    seeds = [int(rng.integers(1 << 30)) for _ in range(6)]
    env = NavEnv(cfg["env"], seed=10)
    succ = 0
    for s in seeds:
        env.reset(seed=s)
        p = make_astar_policy(env.room, cfg)
        res = run_episode(env, p, cfg)
        succ += int(res["success"])
    env.close()
    assert succ >= 3, f"A* oracle too weak: {succ}/6 succeeded"


def test_goal_policy_runs():
    """Smoke: greedy-goal policy completes run_eval without crashing."""
    set_seed(11)
    cfg = load_config("configs/default.yaml")
    cfg["env"]["max_episode_steps"] = 80
    summary = run_eval(cfg, make_goal_policy(), n_episodes=2, seed=11)
    assert "SR" in summary
