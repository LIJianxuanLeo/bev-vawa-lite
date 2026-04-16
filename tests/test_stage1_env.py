"""Stage 1 gate: procedural env + occupancy + A*."""
from __future__ import annotations
import numpy as np
import pytest

from bev_vawa.envs import sample_room, build_xml, rasterize, astar_path, NavEnv
from bev_vawa.envs.occupancy import world_to_cell
from bev_vawa.utils import load_config, set_seed


CFG = load_config("configs/default.yaml")["env"]


def test_room_astar_feasibility():
    """At least 90% of sampled (start, goal) pairs across 10 rooms must be solvable."""
    set_seed(0)
    rng = np.random.default_rng(0)
    feas = 0
    total = 0
    for _ in range(10):
        room = sample_room(rng, CFG)
        grid = rasterize(room, cell_m=CFG["occupancy_cell_m"])
        s = world_to_cell(*room.start, room, CFG["occupancy_cell_m"])
        g = world_to_cell(*room.goal, room, CFG["occupancy_cell_m"])
        path = astar_path(grid, s, g)
        total += 1
        if path is not None and len(path) > 1:
            feas += 1
    assert feas / total >= 0.9, f"feasibility={feas}/{total}"


def test_env_step_and_depth_bounds():
    set_seed(1)
    env = NavEnv(CFG, seed=1)
    obs = env.reset(seed=1)
    assert obs["depth"].shape == tuple(CFG["depth_wh"])[::-1] or obs["depth"].shape == tuple(CFG["depth_wh"])
    depth_min, depth_max = float(obs["depth"].min()), float(obs["depth"].max())
    assert depth_min >= 0.0
    assert depth_max <= CFG["depth_max_m"] + 1e-5

    rng = np.random.default_rng(1)
    for _ in range(50):
        act = rng.uniform(-1, 1, size=2) * np.array([CFG["max_lin_vel"], CFG["max_ang_vel"]])
        step = env.step(act)
        d = step.obs["depth"]
        assert np.isfinite(d).all()
        assert d.min() >= 0.0 and d.max() <= CFG["depth_max_m"] + 1e-5
        assert step.obs["goal_vec"].shape == (2,)
        if step.done:
            break
    env.close()


def test_goal_vector_consistency():
    set_seed(2)
    env = NavEnv(CFG, seed=2)
    obs = env.reset(seed=2)
    x, y, yaw = obs["pose"]
    gx, gy = env.room.goal
    expected_dist = float(np.hypot(gx - x, gy - y))
    assert abs(obs["goal_vec"][0] - expected_dist) < 1e-3
    env.close()
