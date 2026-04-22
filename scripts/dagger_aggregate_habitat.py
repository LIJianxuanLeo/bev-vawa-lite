"""DAGger-style closed-loop data aggregation on Gibson Habitat.

Core DAGger loop for our BEV-VAWA imitation-trained policy:

    1. Load a trained Stage-C checkpoint.
    2. For each Gibson PointNav v2 episode (up to M per scene), reset
       the env, teleport to the episode's start_position, then roll out
       the *policy* for up to T steps. Record (depth, goal, pose) at
       every step into a buffer.
    3. For every buffered state ``(obs_i)`` that has at least H future
       frames available, query the navmesh ``pathfinder`` for the
       shortest path from the agent's current (off-expert-path) position
       to the goal, and derive the expert waypoint + candidate labels
       *as if the agent were teleported there*. This is the
       canonical DAGger correction.
    4. Future frames for the WA ``L_dyn`` / ``L_deadend`` losses are
       taken directly from the policy's own buffered rollout — cheap
       and honest (they reflect what the policy actually does next).
    5. Emit one ``.npz`` shard per scene, schema-v2 compatible with the
       existing ``NavShardDataset`` loader. Mix these shards alongside
       the teleport-expert shards when re-training Stage B / C.

Usage:
    python scripts/dagger_aggregate_habitat.py \
        --config configs/habitat/gibson.yaml \
        --ckpt  /root/data/runs/gibson/stage_c.pt \
        --scene-dir /root/data/scene_datasets/gibson \
        --episode-dir /root/data/datasets/pointnav/gibson/v2 \
        --split train \
        --out /root/data/gibson_dagger_shards/iter1 \
        --max-episodes-per-scene 4 \
        --max-steps-per-episode 150 \
        [--safety]
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch

from bev_vawa.utils import load_config, get_device, set_seed
from bev_vawa.data.expert import candidate_anchors, best_k_for_expert
from bev_vawa.data.rollout_habitat import (
    _label_candidates_habitat,
    _expert_wp_robot_frame,
    _path_resample_xz,
    SHARD_SCHEMA_VERSION,
)
from bev_vawa.models import BEVVAWA
from bev_vawa.eval.policies import make_model_policy, wrap_safety


# ------------------------------------------------------------------
#  Per-rollout sample emission
# ------------------------------------------------------------------

def _emit_samples_from_rollout(
    env,
    anchors: np.ndarray,
    horizon_m: float,
    H: int,
    obs_buffer: List[dict],
    goal_xyz: np.ndarray,
    agent_height: float,
) -> list[dict]:
    """Convert one policy rollout buffer into schema-v2 DAGger samples.

    ``obs_buffer[t]`` is the observation seen by the policy *before* the
    action at step t. We emit a sample for every i where i+H <= len-1,
    i.e. future frames are available. Each sample uses:

    - ``depth``, ``goal``, ``pose`` directly from ``obs_buffer[i]``.
    - ``expert_wp`` from a *fresh* pathfinder shortest path starting at
      ``obs_buffer[i]``'s position (this is the DAGger correction).
    - Candidate labels ``cand_collision`` / ``cand_progress`` /
      ``cand_deadend`` from ``_label_candidates_habitat`` applied at
      the buffered pose.
    - ``future_depth``, ``future_goal`` from ``obs_buffer[i+1..i+H]``
      (i.e.\\ what the policy actually saw after acting).
    """
    pathfinder = env._sim.pathfinder
    samples: list[dict] = []
    N = len(obs_buffer)
    if N < H + 2:
        return samples

    gx = float(goal_xyz[0])
    gy = float(goal_xyz[1])
    gz = float(goal_xyz[2])
    y_world = float(agent_height)  # navmesh works on floor plane, y = agent height

    for i in range(N - H):
        pose = obs_buffer[i]["pose"]          # [x, z, yaw]
        px, pz, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        pos_xyz = np.array([px, y_world, pz], dtype=np.float32)

        # Skip samples that are already essentially at the goal; they
        # contribute nothing to learning and their shortest_path is trivial.
        cur_dist = math.hypot(gx - px, gz - pz)
        if cur_dist < 0.30:
            continue

        # DAGger expert: shortest path from current (off-expert) position
        sp = env.shortest_path(pos_xyz, np.array([gx, gy, gz], dtype=np.float32))
        if sp is None:
            continue
        if sp.geodesic_distance > 20.0:
            continue  # unreachable / pathological
        path_xz = _path_resample_xz(list(sp.points), step_m=0.10)
        if len(path_xz) < 4:
            continue

        pose_xzy = (px, pz, yaw)
        ewp = _expert_wp_robot_frame(path_xz, pose_xzy, horizon_m=horizon_m)
        coll, prog, dead = _label_candidates_habitat(
            env, anchors, pose_xzy,
            np.asarray([gx, gz], dtype=np.float32),
            pathfinder, agent_height,
            want_deadend=True, goal_y=gy,
        )
        best_k = best_k_for_expert(anchors, ewp)

        # Future from the policy's own rollout buffer.
        fd_stack = np.stack([obs_buffer[i + 1 + tau]["depth"]
                             for tau in range(H)]).astype(np.float32)
        fg_stack = np.stack([obs_buffer[i + 1 + tau]["goal"]
                             for tau in range(H)]).astype(np.float32)

        sample_out = {
            "depth": obs_buffer[i]["depth"].astype(np.float32),
            "goal": obs_buffer[i]["goal"].astype(np.float32),
            "pose": obs_buffer[i]["pose"].astype(np.float32),
            "expert_wp": ewp.astype(np.float32),
            "cand_collision": coll.astype(np.float32),
            "cand_progress": prog.astype(np.float32),
            "cand_deadend": (dead if dead is not None
                             else np.zeros_like(coll)).astype(np.float32),
            "best_k": int(best_k),
            "future_depth": fd_stack,
            "future_goal": fg_stack,
        }
        # Carry semantic fields through when the rollout buffer collected
        # them (i.e. env was built with use_semantic=True).
        if "semantic" in obs_buffer[i]:
            sample_out["semantic"] = obs_buffer[i]["semantic"].astype(np.int8)
            fut_sem_stack = np.stack([
                obs_buffer[i + 1 + tau].get(
                    "semantic", np.zeros_like(obs_buffer[i]["semantic"])
                ).astype(np.int8)
                for tau in range(H)
            ])
            sample_out["future_semantic"] = fut_sem_stack
        samples.append(sample_out)
    return samples


# ------------------------------------------------------------------
#  Main driver
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="cfg yaml (typically configs/habitat/gibson.yaml).")
    ap.add_argument("--ckpt", required=True, help="Stage-C checkpoint to roll out.")
    ap.add_argument("--scene-dir", required=True,
                    help="Dir containing Gibson .glb files.")
    ap.add_argument("--episode-dir", required=True,
                    help="pointnav_gibson_v2 dir (contains {train,val}/).")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", required=True, help="Output shard dir.")
    ap.add_argument("--max-episodes-per-scene", type=int, default=4)
    ap.add_argument("--max-steps-per-episode", type=int, default=150)
    ap.add_argument("--scene-limit", type=int, default=None)
    ap.add_argument("--safety", action="store_true",
                    help="wrap rollout policy with reactive-safety (yields "
                         "samples closer to the eval-time distribution when "
                         "safety is on).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="DAGger beta-schedule parameter. Each step, with "
                         "probability beta, the aggregator takes one step "
                         "along the pathfinder's expert path (teleport) "
                         "INSTEAD OF the policy's predicted action. "
                         "beta=1.0 -> pure teleport expert (equivalent to "
                         "rollout_habitat.generate_one_gibson_scene). "
                         "beta=0.0 -> pure policy rollout (original DAGger-1 "
                         "behaviour). Recommended schedule: 0.8 -> 0.4 -> 0 "
                         "across three iterations to smoothly transition "
                         "from on-policy-like to off-policy DAGger.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    va_cfg = cfg["va"]
    K = int(va_cfg["n_candidates"])
    horizon_m = float(va_cfg["waypoint_horizon_m"])
    anchors = candidate_anchors(K, horizon_m)
    H = int(cfg["wa"]["rollout_horizon"])

    # Load model + build rollout policy
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BEVVAWA(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)
    policy = make_model_policy(model, device, cfg, use_wa=True)
    if args.safety:
        policy = wrap_safety(policy, cfg)

    # Deferred Habitat imports
    from bev_vawa.envs.habitat_env import HabitatNavEnv
    from bev_vawa.data.gibson_episodes import iter_episodes, resolve_scene_glb

    # Group episodes by scene name
    episodes_by_scene: dict[str, list[dict]] = {}
    for ep in iter_episodes(args.episode_dir, args.split):
        scene_stem = Path(ep["scene_id"]).stem
        scene_path = os.path.join(args.scene_dir, f"{scene_stem}.glb")
        if not os.path.exists(scene_path):
            continue
        episodes_by_scene.setdefault(scene_stem, []).append(ep)

    scene_names = sorted(episodes_by_scene.keys())
    if args.scene_limit:
        scene_names = scene_names[: args.scene_limit]

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total_samples = 0
    total_rollouts = 0
    total_reached = 0

    for scene_idx, scene_name in enumerate(scene_names):
        scene_path = os.path.join(args.scene_dir, f"{scene_name}.glb")
        episodes = episodes_by_scene[scene_name][: args.max_episodes_per_scene]

        env = HabitatNavEnv(
            cfg["env"], scene_glb=scene_path,
            seed=int(rng.integers(1 << 30)),
        )
        agent_height = env.agent_height

        scene_samples: list[dict] = []
        for ep in episodes:
            goal_xyz = np.asarray(ep["goal_position"], dtype=np.float32)
            start_xyz = np.asarray(ep["start_position"], dtype=np.float32)
            env._goal_xyz = goal_xyz.copy()

            # Seed yaw from start->goal direction so the policy has a
            # roughly reasonable initial heading.
            dx = float(goal_xyz[0] - start_xyz[0])
            dz = float(goal_xyz[2] - start_xyz[2])
            yaw = float(math.atan2(dz, dx)) if (dx * dx + dz * dz) > 1e-8 else 0.0
            env.teleport_xyz(start_xyz, yaw)

            # Reset collision / step counters (episode-scoped)
            env.n_collisions = 0
            env.n_steps = 0
            obs = env._get_obs()

            # Roll out the policy, optionally mixing in expert teleport
            # steps with probability beta (DAGger β-schedule).
            obs_buffer: list[dict] = []
            reached = False
            for step_i in range(args.max_steps_per_episode):
                obs_snapshot = {
                    "depth": obs["depth"].copy(),
                    "goal": obs["goal_vec"].copy(),
                    "pose": obs["pose"].copy(),
                }
                if "semantic" in obs:
                    obs_snapshot["semantic"] = obs["semantic"].copy()
                obs_buffer.append(obs_snapshot)

                use_expert = args.beta > 0.0 and rng.random() < args.beta

                if use_expert:
                    # Query shortest path from current position, teleport
                    # one step along it. If no path, fall through to policy.
                    px, pz, _ = float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2])
                    cur_xyz = np.array([px, agent_height, pz], dtype=np.float32)
                    sp = env.shortest_path(cur_xyz, goal_xyz)
                    if sp is not None and len(sp.points) >= 2:
                        # Step toward the next pathfinder waypoint (clipped
                        # to at most 0.25 m for comparability with discrete
                        # actions) and re-orient to face that direction.
                        p0 = np.asarray(sp.points[0], dtype=np.float32)
                        p1 = np.asarray(sp.points[1], dtype=np.float32)
                        direction = p1 - p0
                        d_norm = float(np.linalg.norm(direction[[0, 2]]))
                        if d_norm > 1e-6:
                            step_m = min(0.25, d_norm)
                            ux = float(direction[0] / d_norm)
                            uz = float(direction[2] / d_norm)
                            new_xyz = np.array([
                                px + ux * step_m, agent_height, pz + uz * step_m,
                            ], dtype=np.float32)
                            new_yaw = float(math.atan2(uz, ux))
                            env.teleport_xyz(new_xyz, new_yaw)
                            env.n_steps += 1
                            obs = env._get_obs()
                            goal_dist = float(obs["goal_vec"][0])
                            if goal_dist < env.goal_tol:
                                reached = True
                                obs_snapshot2 = {
                                    "depth": obs["depth"].copy(),
                                    "goal": obs["goal_vec"].copy(),
                                    "pose": obs["pose"].copy(),
                                }
                                if "semantic" in obs:
                                    obs_snapshot2["semantic"] = obs["semantic"].copy()
                                obs_buffer.append(obs_snapshot2)
                                break
                            continue
                    # fall through if sp was None or degenerate

                v, w = policy(obs, cfg)
                step_out = env.step((v, w))
                obs = step_out.obs
                if step_out.info.get("reached", False):
                    reached = True
                    obs_snapshot2 = {
                        "depth": obs["depth"].copy(),
                        "goal": obs["goal_vec"].copy(),
                        "pose": obs["pose"].copy(),
                    }
                    if "semantic" in obs:
                        obs_snapshot2["semantic"] = obs["semantic"].copy()
                    obs_buffer.append(obs_snapshot2)
                    break
                if step_out.done:
                    obs_snapshot2 = {
                        "depth": obs["depth"].copy(),
                        "goal": obs["goal_vec"].copy(),
                        "pose": obs["pose"].copy(),
                    }
                    if "semantic" in obs:
                        obs_snapshot2["semantic"] = obs["semantic"].copy()
                    obs_buffer.append(obs_snapshot2)
                    break

            total_rollouts += 1
            total_reached += int(reached)
            if len(obs_buffer) < H + 2:
                continue

            ep_samples = _emit_samples_from_rollout(
                env, anchors, horizon_m, H, obs_buffer, goal_xyz, agent_height,
            )
            scene_samples.extend(ep_samples)

        env.close()

        if not scene_samples:
            print(f"[{scene_idx+1}/{len(scene_names)}] scene {scene_name}: "
                  f"0 samples (skipped)")
            continue

        shard = {
            "depth": np.stack([s["depth"] for s in scene_samples]),
            "goal": np.stack([s["goal"] for s in scene_samples]),
            "pose": np.stack([s["pose"] for s in scene_samples]),
            "expert_wp": np.stack([s["expert_wp"] for s in scene_samples]),
            "cand_collision": np.stack([s["cand_collision"] for s in scene_samples]),
            "cand_progress": np.stack([s["cand_progress"] for s in scene_samples]),
            "cand_deadend": np.stack([s["cand_deadend"] for s in scene_samples]),
            "best_k": np.array([s["best_k"] for s in scene_samples], dtype=np.int64),
            "anchors": anchors.astype(np.float32),
            "future_depth": np.stack([s["future_depth"] for s in scene_samples]),
            "future_goal": np.stack([s["future_goal"] for s in scene_samples]),
        }
        # Stamp schema v3 only when every sample in the scene had semantic
        # (keeping NavShardDataset's all-or-nothing v3 detection happy).
        all_have_semantic = all("semantic" in s for s in scene_samples)
        if all_have_semantic:
            shard["semantic"] = np.stack([s["semantic"] for s in scene_samples])
            shard["future_semantic"] = np.stack(
                [s["future_semantic"] for s in scene_samples]
            )
            shard["schema_version"] = np.array(3, dtype=np.int32)
        else:
            shard["schema_version"] = np.array(2, dtype=np.int32)
        out_path = os.path.join(args.out, f"dagger_{scene_name}.npz")
        np.savez_compressed(out_path, **shard)
        total_samples += len(scene_samples)
        print(f"[{scene_idx+1}/{len(scene_names)}] scene {scene_name}: "
              f"{len(scene_samples)} samples -> {out_path}")

    reach_rate = total_reached / max(1, total_rollouts)
    print(
        f"DAGger aggregation done. "
        f"{total_samples} samples across {len(scene_names)} scenes. "
        f"Rollout reach rate = {total_reached}/{total_rollouts} = {reach_rate:.3f}"
    )
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
