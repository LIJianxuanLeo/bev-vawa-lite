"""Policies for closed-loop evaluation.

All policies share the signature ``policy(obs, cfg) -> (v, omega)``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import math
import numpy as np
import torch

from ..control.pure_pursuit import pure_pursuit_cmd
from ..envs.occupancy import rasterize, astar_path, world_to_cell, cell_to_world
from ..envs.pib_generator import RoomSpec


def make_goal_policy():
    """Greedy heading-to-goal baseline (no obstacle avoidance)."""
    def policy(obs, cfg):
        d, bearing = float(obs["goal_vec"][0]), float(obs["goal_vec"][1])
        wp_robot = np.array([d * math.cos(bearing), d * math.sin(bearing)], dtype=np.float32)
        # cap waypoint to a lookahead
        norm = np.linalg.norm(wp_robot)
        if norm > 1.0:
            wp_robot = wp_robot * (1.0 / norm)
        return pure_pursuit_cmd(wp_robot, cfg["env"]["max_lin_vel"], cfg["env"]["max_ang_vel"])
    return policy


def make_astar_policy(room: RoomSpec, cfg: dict):
    """Upper-bound oracle: re-plan A* on the ground-truth occupancy grid every step,
    grab the ~1m-ahead waypoint, apply pure-pursuit.
    Uses extra inflation (robot_radius + 0.15m) so the planner leaves margin for
    the controller's imperfect tracking."""
    cell_m = cfg["env"]["occupancy_cell_m"]
    inflate = room.robot_radius + 0.05
    grid = rasterize(room, cell_m=cell_m, inflate_m=inflate)
    horizon = cfg["va"]["waypoint_horizon_m"]
    def _free_cell(grid_, rc):
        """Return rc if free else the nearest free cell within a small radius."""
        if 0 <= rc[0] < grid_.shape[0] and 0 <= rc[1] < grid_.shape[1] and not grid_[rc]:
            return rc
        for r_ in range(1, 8):
            for dr in range(-r_, r_ + 1):
                for dc in range(-r_, r_ + 1):
                    nr, nc = rc[0] + dr, rc[1] + dc
                    if 0 <= nr < grid_.shape[0] and 0 <= nc < grid_.shape[1] and not grid_[nr, nc]:
                        return (nr, nc)
        return rc

    def policy(obs, cfg):
        x, y, yaw = obs["pose"]
        s = world_to_cell(x, y, room, cell_m)
        g = world_to_cell(*room.goal, room, cell_m)
        grid_local = grid
        if (0 <= s[0] < grid.shape[0] and 0 <= s[1] < grid.shape[1] and grid[s]) or \
           (0 <= g[0] < grid.shape[0] and 0 <= g[1] < grid.shape[1] and grid[g]):
            grid_local = grid.copy()
            s = _free_cell(grid_local, s)
            g = _free_cell(grid_local, g)
        path = astar_path(grid_local, s, g)
        if path is None or len(path) < 2:
            return 0.0, 0.0
        # walk horizon along the path
        acc = 0.0
        tgt = path[-1]
        for i in range(len(path) - 1):
            x0, y0 = cell_to_world(*path[i], room, cell_m)
            x1, y1 = cell_to_world(*path[i + 1], room, cell_m)
            acc += math.hypot(x1 - x0, y1 - y0)
            if acc >= horizon:
                tgt = path[i + 1]
                break
        tx, ty = cell_to_world(*tgt, room, cell_m)
        dx, dy = tx - x, ty - y
        cy_, sy_ = math.cos(-yaw), math.sin(-yaw)
        rx = cy_ * dx - sy_ * dy
        ry = sy_ * dx + cy_ * dy
        return pure_pursuit_cmd((rx, ry), cfg["env"]["max_lin_vel"], cfg["env"]["max_ang_vel"])
    return policy


def make_model_policy(model, device, cfg: dict, use_wa: bool = True):
    """Wrap a trained BEVVAWA in a (v, omega) callable."""
    model.eval()
    def policy(obs, cfg):
        depth = torch.from_numpy(obs["depth"]).to(device).unsqueeze(0).unsqueeze(0) / cfg["env"]["depth_max_m"]
        goal = torch.from_numpy(obs["goal_vec"]).to(device).unsqueeze(0)
        with torch.no_grad():
            out = model(depth, goal, use_wa=use_wa)
            wp, k = model.select_waypoint(out)
        wp_rf = wp[0].cpu().numpy()
        return pure_pursuit_cmd(wp_rf, cfg["env"]["max_lin_vel"], cfg["env"]["max_ang_vel"])
    return policy


def load_model_policy(ckpt_path: str, cfg: dict, device, use_wa: bool = True):
    from ..models import BEVVAWA
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BEVVAWA(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    return make_model_policy(model, device, cfg, use_wa=use_wa)
