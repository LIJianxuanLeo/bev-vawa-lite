"""Expert-label utilities: A* path smoothing, candidate anchor fan, labeling.

Candidate anchors are K local waypoints sampled on a fan around the robot
heading at a fixed horizon (``waypoint_horizon_m``). The VA branch scores
them; the WA branch predicts risk/progress per anchor; labels come from the
occupancy grid + expert path geometry.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

from ..envs.occupancy import world_to_cell
from ..envs.pib_generator import RoomSpec


def chaikin_smooth(path: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner cutting on an (N, 2) polyline."""
    pts = np.asarray(path, dtype=np.float32)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        out.append(pts[-1])
        pts = np.asarray(out, dtype=np.float32)
    return pts


def path_resample(path: np.ndarray, step_m: float) -> np.ndarray:
    """Arc-length resample the polyline so consecutive points are ~step_m apart."""
    pts = np.asarray(path, dtype=np.float32)
    if len(pts) < 2:
        return pts
    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total = float(seg_lens.sum())
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(np.ceil(total / step_m)) + 1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    targets = np.linspace(0.0, total, n)
    out_x = np.interp(targets, cum, pts[:, 0])
    out_y = np.interp(targets, cum, pts[:, 1])
    return np.stack([out_x, out_y], axis=1).astype(np.float32)


def expert_waypoint_from_path(path_world: np.ndarray, pose: Tuple[float, float, float], horizon_m: float) -> np.ndarray:
    """Return the expert next-waypoint in **robot frame** (x-fwd, y-left), 2-vector."""
    x, y, yaw = pose
    d2 = np.sum((path_world - np.asarray([x, y])[None, :]) ** 2, axis=1)
    idx_near = int(np.argmin(d2))
    # walk forward until we exceed horizon
    target_idx = idx_near
    acc = 0.0
    for i in range(idx_near, len(path_world) - 1):
        acc += float(np.linalg.norm(path_world[i + 1] - path_world[i]))
        target_idx = i + 1
        if acc >= horizon_m:
            break
    tgt = path_world[target_idx]
    dx, dy = tgt[0] - x, tgt[1] - y
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    rx = cy * dx - sy * dy
    ry = sy * dx + cy * dy
    return np.asarray([rx, ry], dtype=np.float32)


def candidate_anchors(K: int, horizon_m: float, fan_deg: float = 120.0) -> np.ndarray:
    """Return (K, 2) anchor points in robot frame on a fan in front of the robot."""
    angles = np.deg2rad(np.linspace(-fan_deg / 2, fan_deg / 2, K))
    xs = horizon_m * np.cos(angles)
    ys = horizon_m * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _segment_cells(r0, c0, r1, c1):
    """Integer raster of a line segment (Bresenham-ish, unbounded)."""
    dr = r1 - r0
    dc = c1 - c0
    n = max(abs(dr), abs(dc)) + 1
    rs = np.linspace(r0, r1, n)
    cs = np.linspace(c0, c1, n)
    return np.stack([np.rint(rs).astype(int), np.rint(cs).astype(int)], axis=1)


def label_candidates(
    anchors_robot: np.ndarray,
    pose: Tuple[float, float, float],
    goal_xy: Tuple[float, float],
    grid: np.ndarray,
    room: RoomSpec,
    cell_m: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """For each anchor, compute (collision_bool, progress_float) and return also
    the index of the anchor closest to the expert direction (= best-of-K target).

    collision = any cell along the straight segment (pose -> world(anchor)) is occupied
    progress  = distance(pose, goal) - distance(anchor_world, goal)        (meters)
    best_k    = argmin over k of euclidean distance between anchor_world_k
                and (the point on the A* path ~ horizon away). Computed outside
                with expert_waypoint_from_path; we expose a helper that just
                returns the nearest-anchor index given the expert waypoint.
    """
    x, y, yaw = pose
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    H, W = grid.shape
    K = anchors_robot.shape[0]
    collisions = np.zeros(K, dtype=np.bool_)
    progress = np.zeros(K, dtype=np.float32)

    start_rc = world_to_cell(x, y, room, cell_m)
    gx, gy = goal_xy
    start_dist = float(np.hypot(gx - x, gy - y))

    for k in range(K):
        ax_r, ay_r = anchors_robot[k]
        # robot -> world
        ax = x + cy_ * ax_r - sy_ * ay_r
        ay = y + sy_ * ax_r + cy_ * ay_r
        end_rc = world_to_cell(ax, ay, room, cell_m)
        cells = _segment_cells(start_rc[0], start_rc[1], end_rc[0], end_rc[1])
        in_bounds = (cells[:, 0] >= 0) & (cells[:, 0] < H) & (cells[:, 1] >= 0) & (cells[:, 1] < W)
        if not in_bounds.all():
            collisions[k] = True
        else:
            collisions[k] = bool(grid[cells[:, 0], cells[:, 1]].any())
        progress[k] = start_dist - float(np.hypot(gx - ax, gy - ay))
    return collisions, progress, -1  # best_k resolved by caller using expert_waypoint


def best_k_for_expert(anchors_robot: np.ndarray, expert_wp_robot: np.ndarray) -> int:
    d2 = np.sum((anchors_robot - expert_wp_robot[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))
