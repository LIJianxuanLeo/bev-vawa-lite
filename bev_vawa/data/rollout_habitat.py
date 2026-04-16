"""Habitat-sim offline dataset generator.

Produces .npz shards with **exactly the same schema** as
``bev_vawa.data.rollout.generate_dataset`` (the MuJoCo version), so downstream
training (``scripts/train.py``) consumes either source identically.

Pipeline per scene:

    1. Sample M start-goal pairs that are both navigable and 2-15 m apart.
    2. For each pair, get the geodesic shortest path from habitat's pathfinder.
    3. Walk the path in 0.10 m steps.
    4. At each step, teleport the agent, render depth, and compute labels:
        * expert_wp   — the path point 1.5 m ahead, in robot frame
        * best_k      — arg-min over K anchor directions
        * cand_collision(k) — does a 1.5 m straight raycast from the agent
          along anchor k collide with the navmesh before H*0.5 m?
        * cand_progress(k)  — Δ distance-to-goal if we advanced 0.5 m along
          that anchor direction.

This is intentionally lightweight — we do NOT roll out each candidate in
habitat's physics. The cheap raycast/geodesic heuristic matches what the
MuJoCo version does on its occupancy grid, and it keeps data generation fast
(embarrassingly parallel across scenes on the remote box).
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import List, Optional
import numpy as np

from ..data.expert import candidate_anchors, best_k_for_expert
from ..utils import get_logger

log = get_logger(__name__)


def _path_resample_xz(points: List[np.ndarray], step_m: float) -> np.ndarray:
    """Arc-length resample a sequence of 3-D habitat path points (Y-up) onto
    a uniform grid on the X-Z floor plane, returning shape (N, 3)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) < 2:
        return pts
    segs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(segs[:, [0, 2]], axis=1)  # floor-plane distance
    total = float(seg_len.sum())
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(math.ceil(total / step_m)) + 1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    tgt = np.linspace(0.0, total, n)
    out = np.stack([
        np.interp(tgt, cum, pts[:, 0]),
        np.interp(tgt, cum, pts[:, 1]),
        np.interp(tgt, cum, pts[:, 2]),
    ], axis=1).astype(np.float32)
    return out


def _expert_wp_robot_frame(path_xz: np.ndarray, pose_xzy: tuple, horizon_m: float) -> np.ndarray:
    """Return 2-D expert waypoint in robot frame (x-fwd, y-left)."""
    x, z, yaw = pose_xzy
    d2 = (path_xz[:, 0] - x) ** 2 + (path_xz[:, 2] - z) ** 2
    i0 = int(np.argmin(d2))
    acc = 0.0
    target_idx = i0
    for i in range(i0, len(path_xz) - 1):
        dx = path_xz[i + 1, 0] - path_xz[i, 0]
        dz = path_xz[i + 1, 2] - path_xz[i, 2]
        acc += float(math.hypot(dx, dz))
        target_idx = i + 1
        if acc >= horizon_m:
            break
    tx, tz = path_xz[target_idx, 0], path_xz[target_idx, 2]
    dx, dz = tx - x, tz - z
    c, s = math.cos(-yaw), math.sin(-yaw)
    rx = c * dx - s * dz
    ry = s * dx + c * dz
    return np.asarray([rx, ry], dtype=np.float32)


def _label_candidates_habitat(
    env,
    anchors: np.ndarray,          # (K, 2) robot-frame (x-fwd, y-left)
    pose_xzy: tuple,              # (x, z, yaw)
    goal_xz: np.ndarray,          # (2,) world (x, z)
    pathfinder,
    agent_height: float,
    step_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cand_collision, cand_progress) of shape (K,) each.

    cand_collision[k] = 1 if the navmesh blocks a straight step of ``step_m``
    from the current pose along anchor k's direction (used as a cheap proxy
    for near-future collision risk).

    cand_progress[k] = (current_goal_dist − new_goal_dist), where new_goal_dist
    is the geodesic distance from the post-step position to the goal.
    """
    x, z, yaw = pose_xzy
    gx, gz = float(goal_xz[0]), float(goal_xz[1])
    cur_dist = float(math.hypot(gx - x, gz - z))

    K = anchors.shape[0]
    coll = np.zeros(K, dtype=np.float32)
    prog = np.zeros(K, dtype=np.float32)

    # Pre-compute world-frame step directions for each anchor. Anchor vectors
    # are expressed in robot frame, so rotate by yaw into world (x, z).
    c, s = math.cos(yaw), math.sin(yaw)
    for k in range(K):
        ax, ay = float(anchors[k, 0]), float(anchors[k, 1])
        wx = c * ax - s * ay
        wz = s * ax + c * ay
        norm = math.hypot(wx, wz)
        if norm < 1e-6:
            continue
        ux, uz = wx / norm, wz / norm
        p_from = np.array([x, agent_height, z], dtype=np.float32)
        p_to = np.array([x + ux * step_m, agent_height, z + uz * step_m], dtype=np.float32)
        stepped = pathfinder.try_step(p_from, p_to)
        gap = float(np.linalg.norm(np.asarray(stepped) - p_to))
        coll[k] = 1.0 if gap > 1e-3 else 0.0

        # progress = geodesic goal-distance delta
        sp = env.shortest_path(np.asarray(stepped),
                               np.array([gx, agent_height, gz], dtype=np.float32))
        new_dist = float(sp.geodesic_distance) if sp is not None else cur_dist
        prog[k] = float(cur_dist - new_dist)

    # normalise progress to ~[-1, 1]
    prog = np.clip(prog / max(step_m, 1e-3), -2.0, 2.0)
    return coll, prog


def generate_one_scene(
    env,                           # HabitatNavEnv instance
    cfg: dict,
    n_pairs: int,
    samples_per_pair: int,
    seed: int = 0,
) -> Optional[dict]:
    """Sample episodes inside one scene; collect training tuples along each
    shortest path. Returns a dict with the MuJoCo-compatible keys or None if
    the scene has no solvable pairs."""
    import habitat_sim  # safe — we only reach here on the remote box
    rng = np.random.default_rng(seed)
    va_cfg = cfg["va"]
    env_cfg = cfg["env"]
    K = int(va_cfg["n_candidates"])
    horizon = float(va_cfg["waypoint_horizon_m"])
    anchors = candidate_anchors(K, horizon)

    pathfinder = env._sim.pathfinder
    agent_height = env.agent_height

    depths, goals, poses = [], [], []
    expert_wps, cand_coll, cand_prog, best_ks = [], [], [], []

    solved_pairs = 0
    tries = 0
    while solved_pairs < n_pairs and tries < n_pairs * 5:
        tries += 1
        s = pathfinder.get_random_navigable_point()
        g = pathfinder.get_random_navigable_point()
        sp = env.shortest_path(np.asarray(s), np.asarray(g))
        if sp is None or sp.geodesic_distance < 2.0 or sp.geodesic_distance > 15.0:
            continue
        path_xz = _path_resample_xz(list(sp.points), step_m=0.10)
        if len(path_xz) < 4:
            continue

        idx_pool = np.arange(len(path_xz) - 1)
        picks = rng.choice(idx_pool, size=min(samples_per_pair, len(idx_pool)),
                           replace=len(idx_pool) < samples_per_pair)

        for i in picks:
            base = path_xz[i]
            nxt = path_xz[min(i + 2, len(path_xz) - 1)]
            dx, dz = nxt[0] - base[0], nxt[2] - base[2]
            if math.hypot(dx, dz) < 1e-6:
                continue
            yaw = float(math.atan2(dz, dx))
            yaw += float(rng.normal(0, 0.15))
            px = float(base[0] + rng.normal(0, 0.05))
            pz = float(base[2] + rng.normal(0, 0.05))
            pos_xyz = np.array([px, base[1], pz], dtype=np.float32)
            env.teleport_xyz(pos_xyz, yaw)
            # set env's goal for obs computation to match sample's (start, goal)
            env._goal_xyz = np.asarray([g[0], g[1], g[2]], dtype=np.float32)
            obs = env._get_obs()
            pose_xzy = (px, pz, yaw)

            ewp = _expert_wp_robot_frame(path_xz, pose_xzy, horizon_m=horizon)
            coll, prog = _label_candidates_habitat(
                env, anchors, pose_xzy, np.asarray([g[0], g[2]], dtype=np.float32),
                pathfinder, agent_height,
            )
            best_k = best_k_for_expert(anchors, ewp)

            depths.append(obs["depth"].astype(np.float32))
            goals.append(obs["goal_vec"].astype(np.float32))
            poses.append(obs["pose"].astype(np.float32))
            expert_wps.append(ewp.astype(np.float32))
            cand_coll.append(coll.astype(np.float32))
            cand_prog.append(prog.astype(np.float32))
            best_ks.append(np.int64(best_k))
        solved_pairs += 1

    if not depths:
        return None
    return {
        "depth": np.stack(depths),
        "goal": np.stack(goals),
        "pose": np.stack(poses),
        "expert_wp": np.stack(expert_wps),
        "cand_collision": np.stack(cand_coll),
        "cand_progress": np.stack(cand_prog),
        "best_k": np.stack(best_ks),
        "anchors": anchors.astype(np.float32),
    }


def generate_dataset_habitat(
    cfg: dict,
    scenes: List[str],
    out_dir: str,
    n_pairs_per_scene: int = 8,
    samples_per_pair: int = 8,
    seed: int = 0,
    gpu_device_id: int = 0,
) -> int:
    """Iterate scene .glb files and dump one .npz shard per scene."""
    from .habitat_env import HabitatNavEnv   # local import is fine — caller ensures habitat-sim present
    # NOTE: the import above will fail cleanly on macOS; this file is only meant
    # to run inside the docker/habitat image.
    raise_if_no_habitat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for scene_i, scene in enumerate(scenes):
        try:
            env = HabitatNavEnv(cfg["env"], scene_glb=scene, seed=seed + scene_i,
                                gpu_device_id=gpu_device_id)
        except Exception as e:
            log.warning(f"skipping scene {scene}: init failed ({e})")
            continue
        shard = generate_one_scene(env, cfg, n_pairs_per_scene, samples_per_pair,
                                    seed=seed + scene_i)
        env.close()
        if shard is None:
            log.warning(f"no samples for scene {scene}")
            continue
        stem = Path(scene).stem
        p = out / f"scene_{stem}.npz"
        np.savez_compressed(p, **shard)
        written += 1
        if (scene_i + 1) % 5 == 0:
            log.info(f"[{scene_i+1}/{len(scenes)}] scenes, {written} shards written to {out}")
    log.info(f"Done. {written} shards in {out}")
    return written


def raise_if_no_habitat() -> None:
    try:
        import habitat_sim  # noqa: F401
    except Exception as e:
        raise ImportError(
            "habitat-sim is not available. Use docker/habitat.Dockerfile on a "
            f"Linux GPU host. Original error: {e!r}"
        )
