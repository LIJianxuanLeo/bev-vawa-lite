"""Policies for closed-loop evaluation.

All policies share the signature ``policy(obs, cfg) -> (v, omega)``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import math
import numpy as np
import torch

from ..control.pure_pursuit import pure_pursuit_cmd
from ..envs.occupancy import rasterize, astar_path, world_to_cell, cell_to_world
from ..envs.pib_generator import RoomSpec


# ---------------------------------------------------------------------------
# Reactive safety wrapper (method-agnostic)
# ---------------------------------------------------------------------------
def wrap_safety(policy: Callable, cfg: dict) -> Callable:
    """Compose a learned policy with a reactive obstacle-avoidance override.

    v3 — APF on forward arc + **lateral-sector scrape guard**.

    Additions over v2:
      * The outer image columns (beyond the forward arc, i.e. roughly
        ±27°..±45° at FOV=90°) are monitored independently. When a side
        clearance falls below ``side_warn_m`` the wrapper adds an additive
        ω push *away* from that side, scaled by how close the obstacle is.
        This catches the side-scrape failure mode that v2 is blind to
        (forward appears clear, but robot's shoulder is already grazing
        an obstacle during a turn).

    Per-band behaviour for the forward arc (unchanged from v2):

        d_min >= warn_m   →  cruise, pass (v, ω) through unchanged
        near_m <= d_min < warn_m
                          →  keep v, ω' = ω + asym * max_ω * (0.5..1.0)
        d_min <  near_m   →  v' = max_lin * near_forward_frac,
                              strong corrective ω (sign from asym, fallback
                              to policy ω or "turn left" when symmetric)

    Side correction is **always** computed (even when forward is clear) and
    added after the forward correction, then the total ω is clipped.

    Knobs live under ``cfg['safety']``. Invariant: wrapper never increases
    |v| or |ω| beyond env limits.
    """
    sc = cfg.get("safety", {}) or {}
    near_m = float(sc.get("near_m", 0.35))
    warn_m = float(sc.get("warn_m", 0.60))
    arc_frac = float(sc.get("forward_arc_frac", 0.60))      # ±27° at FOV=90°
    row_lo, row_hi = sc.get("row_frac", [0.3, 0.7])
    near_forward_frac = float(sc.get("near_forward_frac", 0.30))
    asym_tiebreak = float(sc.get("asym_tiebreak", 0.15))
    # v3: side sector knobs. side_warn_m is tighter than forward warn_m
    # because sides become a collision risk only when very close.
    side_warn_m = float(sc.get("side_warn_m", 0.40))
    side_w_gain = float(sc.get("side_w_gain", 0.6))         # × max_w
    side_v_taper = float(sc.get("side_v_taper", 0.85))      # v multiplier when side close
    max_v = float(cfg["env"]["max_lin_vel"])
    max_w = float(cfg["env"]["max_ang_vel"])
    min_valid_m = 0.05

    def _side_near(x: np.ndarray, fallback: float = None) -> float:
        """Robust near-clearance for a depth patch: 5th percentile of
        valid pixels (ignoring sensor-miss zeros). Falls back to
        ``fallback`` (default: ``warn_m``) if the patch is empty — i.e.
        treated as clear."""
        if fallback is None:
            fallback = warn_m
        valid = x > min_valid_m
        if valid.sum() < 10:
            return fallback
        return float(np.percentile(x[valid], 5))

    def _closeness(d: float, warn: float) -> float:
        """Map a clearance distance to a [0, 1] closeness score.
        0 = clear (d >= warn), 1 = touching (d <= 0)."""
        if d >= warn:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - d / max(warn, 1e-6))))

    def safe_policy(obs, cfg_):
        v, w = policy(obs, cfg_)
        depth = obs["depth"]
        if depth is None or depth.size == 0:
            return v, w
        H, W = depth.shape
        r0, r1 = int(H * row_lo), int(H * row_hi)
        c_lo = int(W * (0.5 - arc_frac / 2.0))
        c_hi = int(W * (0.5 + arc_frac / 2.0))

        arc = depth[r0:r1, c_lo:c_hi]
        # v3: outer image columns = side sectors (convention here matches
        # the existing forward-arc asym: lower col index = robot LEFT).
        left_side = depth[r0:r1, :c_lo] if c_lo > 0 else None
        right_side = depth[r0:r1, c_hi:] if c_hi < W else None

        d_left = _side_near(left_side, side_warn_m) if left_side is not None and left_side.size > 0 else side_warn_m
        d_right = _side_near(right_side, side_warn_m) if right_side is not None and right_side.size > 0 else side_warn_m
        cl_left = _closeness(d_left, side_warn_m)
        cl_right = _closeness(d_right, side_warn_m)
        # obstacle on LEFT (cl_left high) → steer RIGHT (-ω); sign matches
        # forward-arc convention (+ω = left).
        w_side = (cl_right - cl_left) * side_w_gain * max_w
        # gentle v taper when either side is close (robot is threading a gap)
        side_pressure = max(cl_left, cl_right)
        v_side_mult = 1.0 - (1.0 - side_v_taper) * side_pressure

        # --- forward arc (v2 logic) ---
        valid = arc > min_valid_m
        if not valid.any():
            # forward blind (all sensor misses): still apply side correction
            w_new = float(np.clip(w + w_side, -max_w, max_w))
            return float(v) * v_side_mult, w_new

        d_min = float(arc[valid].min())
        if d_min >= warn_m:
            # cruise, but side correction still active
            w_new = float(np.clip(w + w_side, -max_w, max_w))
            return float(v) * v_side_mult, w_new

        mid = max(1, arc.shape[1] // 2)
        l_near = _side_near(arc[:, :mid])
        r_near = _side_near(arc[:, mid:])
        asym = (l_near - r_near) / max(warn_m, 1e-6)
        asym = float(np.clip(asym, -1.0, 1.0))
        urgency = max(0.0, min(1.0, (warn_m - d_min) / max(warn_m - near_m, 1e-6)))
        w_corr = asym * max_w * (0.5 + 0.5 * urgency)

        if d_min < near_m:
            if abs(asym) < asym_tiebreak:
                direction = 1.0 if w >= 0 else -1.0
                if abs(w) < 0.05:
                    direction = 1.0
                w_corr = direction * max_w * 0.75
            else:
                w_corr = float(np.sign(asym)) * max_w * 0.9
            v_new = max_v * near_forward_frac
        else:
            v_new = float(v)                                 # warn band: keep v

        # Combine forward + side corrections; side taper applies multiplicatively
        # to the already-computed v_new so we slow slightly further in pinch points.
        v_new = v_new * v_side_mult
        w_new = float(np.clip(w + w_corr + w_side, -max_w, max_w))
        return v_new, w_new

    return safe_policy


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
    """Wrap a trained BEVVAWA in a (v, omega) callable.

    Automatically forwards the ``semantic`` observation when (a) the env
    produces one and (b) the model was built with ``bev.use_semantic=True``.
    Gracefully no-ops on depth-only setups so the same wrapper works on
    PIB-Nav, Gibson-without-semantic, and HM3D-with-semantic.
    """
    model.eval()
    expects_semantic = bool(cfg.get("bev", {}).get("use_semantic", False))
    def policy(obs, cfg):
        depth = torch.from_numpy(obs["depth"]).to(device).unsqueeze(0).unsqueeze(0) / cfg["env"]["depth_max_m"]
        goal = torch.from_numpy(obs["goal_vec"]).to(device).unsqueeze(0)
        sem_tensor = None
        if expects_semantic and "semantic" in obs:
            sem_tensor = torch.from_numpy(obs["semantic"]).to(device).long().unsqueeze(0)
        with torch.no_grad():
            out = model(depth, goal, use_wa=use_wa, semantic=sem_tensor)
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
