"""Closed-loop visualization demo for a trained BEV-VAWA checkpoint.

Runs N PIB-Nav episodes with a trained model and writes an MP4 per episode
(and optionally a single concatenated reel). Each frame shows a 4-panel view:

  | top-down RGB (scene)    | BEV schematic (traj + obstacles) |
  | first-person RGB        | depth heatmap                    |

The top-down RGB is a pure MuJoCo render of the scene (no overlays — because
the camera azimuth / near-vertical elevation makes a world→image transform
non-trivial), and the BEV schematic is a synthetic matplotlib plot drawn in
world coordinates so every marker is guaranteed to line up with the scene
geometry (room rectangle, obstacle boxes, start / goal / current pose).

Text overlay on top: episode #, step, collisions, goal distance, status.

Usage:
    PYTHONPATH=$PWD python scripts/viz_demo.py \\
        --config configs/default.yaml \\
        --ckpt   runs/default/stage_c.pt \\
        --n-episodes 5 --out results/demo

Outputs:
    results/demo/ep{000..N-1}.mp4
    results/demo/reel.mp4              (if --reel)
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
from typing import List

import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from bev_vawa.utils import load_config, set_seed, get_device
from bev_vawa.envs.mujoco_env import NavEnv
from bev_vawa.envs.pib_generator import build_xml
from bev_vawa.eval.policies import load_model_policy, wrap_safety


VIEW_WH = (640, 480)   # viewer / top-down size
FPV_WH = (320, 240)    # first-person RGB size


def _enlarge_framebuffer(env: NavEnv, render_wh=(640, 480)) -> None:
    """Rebuild env's MuJoCo model with a larger offscreen framebuffer so we can
    spawn RGB renderers larger than the depth-sensor size. The depth renderer
    (128x128 in this project) still works — offwidth/offheight is a MAX.

    Preserves sim state (qpos/qvel) and the env's own depth renderer.
    """
    xml = build_xml(env.room, depth_wh=render_wh, fov_deg=env.fov)
    new_model = mujoco.MjModel.from_xml_string(xml)
    new_data = mujoco.MjData(new_model)
    # copy current sim state across (joint layout is identical)
    new_data.qpos[:] = env.data.qpos[:]
    new_data.qvel[:] = env.data.qvel[:]
    mujoco.mj_forward(new_model, new_data)
    env.model = new_model
    env.data = new_data
    # recreate env's depth renderer bound to the new model
    if env._renderer is not None:
        try:
            env._renderer.close()
        except Exception:
            pass
    env._renderer = mujoco.Renderer(
        env.model, height=env.depth_wh[1], width=env.depth_wh[0]
    )
    env._renderer.enable_depth_rendering()


def _build_rgb_renderer(model, width: int, height: int) -> mujoco.Renderer:
    """RGB-mode renderer bound to a model. Independent of the env's depth renderer."""
    r = mujoco.Renderer(model, height=height, width=width)
    # no enable_depth_rendering() → RGB by default
    return r


def _room_bounds(room):
    """RoomSpec stores width/depth only; the floor is centered at origin."""
    x_min = -0.5 * room.width
    x_max = +0.5 * room.width
    y_min = -0.5 * room.depth
    y_max = +0.5 * room.depth
    return x_min, x_max, y_min, y_max


def _topdown_camera(room) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    x_min, x_max, y_min, y_max = _room_bounds(room)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cam.lookat = np.array([cx, cy, 0.0])
    extent = max(x_max - x_min, y_max - y_min)
    cam.distance = 1.15 * extent
    cam.elevation = -89.0   # almost straight down (MuJoCo clamps at -90)
    cam.azimuth = 90.0
    return cam


def _fpv_camera() -> mujoco.MjvCamera:
    """Render from the robot's mounted cam_depth camera in RGB mode."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    # fixedcamid will be set per-model after we know which index cam_depth is
    return cam


def _render_topdown(renderer: mujoco.Renderer, data, cam) -> np.ndarray:
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    return np.asarray(rgb, dtype=np.uint8)


def _render_fpv(renderer: mujoco.Renderer, data) -> np.ndarray:
    renderer.update_scene(data, camera="cam_depth")
    rgb = renderer.render()
    return np.asarray(rgb, dtype=np.uint8)


def _draw_bev_schematic(ax, room, traj: List[tuple]) -> None:
    """Synthetic BEV: room rectangle + obstacles + trajectory + markers.
    Drawn entirely in world coordinates so overlays are guaranteed to align."""
    x_min, x_max, y_min, y_max = _room_bounds(room)
    # light floor + wall rectangle
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               facecolor="#f2f2f2", edgecolor="black",
                               linewidth=2.0, zorder=0))
    # obstacle boxes
    for o in room.obstacles:
        ax.add_patch(plt.Rectangle((o.cx - o.sx, o.cy - o.sy),
                                   2.0 * o.sx, 2.0 * o.sy,
                                   facecolor="#8a8a8a", edgecolor="black",
                                   linewidth=1.0, zorder=1))
    # trajectory
    if len(traj) >= 2:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, "-", color="tab:cyan", lw=2.2, alpha=0.95, zorder=2)
    # start (white circle)
    ax.plot(room.start[0], room.start[1], "o", color="white",
            markersize=9, markeredgecolor="black", zorder=3)
    # goal (gold star)
    ax.plot(room.goal[0], room.goal[1], "*", color="gold",
            markersize=16, markeredgecolor="black", zorder=3)
    # current robot position + heading arrow
    if traj:
        x, y, yaw = traj[-1]
        ax.plot(x, y, "o", color="red", markersize=9,
                markeredgecolor="black", zorder=4)
        r = 0.35
        ax.annotate("",
                    xy=(x + r * np.cos(yaw), y + r * np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.8),
                    zorder=4)

    pad = 0.25
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.set_title("BEV schematic (world coords)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def _draw_frame(
    top_rgb: np.ndarray,
    fpv_rgb: np.ndarray,
    depth: np.ndarray,
    depth_max: float,
    traj: List[tuple],
    room,
    ep_idx: int,
    step_i: int,
    n_collisions: int,
    goal_dist: float,
    status: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(13, 7), dpi=100)
    gs = fig.add_gridspec(2, 2,
                          left=0.03, right=0.98, top=0.91, bottom=0.04,
                          wspace=0.08, hspace=0.14)

    # --- top-down RGB (pure MuJoCo render, no overlays) ---
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.imshow(top_rgb)
    ax_top.set_title("Top-down RGB (scene)", fontsize=10)
    ax_top.axis("off")

    # --- BEV schematic (world coords) ---
    ax_bev = fig.add_subplot(gs[0, 1])
    _draw_bev_schematic(ax_bev, room, traj)

    # --- first-person RGB ---
    ax_fpv = fig.add_subplot(gs[1, 0])
    ax_fpv.imshow(fpv_rgb)
    ax_fpv.set_title("First-person (robot cam)", fontsize=10)
    ax_fpv.axis("off")

    # --- depth heatmap ---
    ax_dep = fig.add_subplot(gs[1, 1])
    im = ax_dep.imshow(depth, cmap="turbo", vmin=0.0, vmax=depth_max)
    ax_dep.set_title("Depth (m)", fontsize=10)
    ax_dep.axis("off")

    # --- top title bar ---
    color = {"running": "black", "success": "green",
             "collision": "red", "timeout": "orange"}.get(status, "black")
    fig.suptitle(
        f"Episode {ep_idx}  |  step {step_i:3d}  |  collisions {n_collisions:2d}  |  "
        f"goal dist {goal_dist:.2f} m  |  {status.upper()}",
        fontsize=12, color=color,
    )

    # --- render to numpy ---
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def _run_one_episode(
    env: NavEnv, policy, cfg: dict, ep_idx: int, out_path: Path,
    max_steps: int,
) -> dict:
    # MuJoCo offscreen framebuffer defaults to depth_wh (128x128) from
    # build_xml → too small for our 640x480 top-down renderer. Rebuild the
    # model with a larger offwidth/offheight; sim state is preserved.
    render_wh = (max(VIEW_WH[0], FPV_WH[0]), max(VIEW_WH[1], FPV_WH[1]))
    _enlarge_framebuffer(env, render_wh=render_wh)

    # build independent RGB renderers (env's renderer is in depth mode)
    topdown_r = _build_rgb_renderer(env.model, VIEW_WH[0], VIEW_WH[1])
    fpv_r = _build_rgb_renderer(env.model, FPV_WH[0], FPV_WH[1])
    topdown_cam = _topdown_camera(env.room)

    traj: List[tuple] = []
    obs = env._get_obs()
    traj.append((float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2])))

    writer = imageio.get_writer(str(out_path), fps=10, codec="libx264",
                                quality=8, macro_block_size=1)
    status = "running"
    n_steps = 0
    try:
        for step_i in range(max_steps):
            top_rgb = _render_topdown(topdown_r, env.data, topdown_cam)
            fpv_rgb = _render_fpv(fpv_r, env.data)
            frame = _draw_frame(
                top_rgb, fpv_rgb, obs["depth"], cfg["env"]["depth_max_m"],
                traj, env.room, ep_idx, step_i, env.n_collisions,
                float(obs["goal_vec"][0]), status,
            )
            writer.append_data(frame)

            v, w = policy(obs, cfg)
            step = env.step((v, w))
            obs = step.obs
            traj.append((float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2])))
            n_steps += 1
            if step.info["reached"]:
                status = "success"; break
            if step.info["n_collisions"] >= env.max_collisions:
                status = "collision"; break
            if step.info["timeout"]:
                status = "timeout"; break
            if step.done:
                status = "collision" if env.n_collisions >= env.max_collisions else "timeout"
                break

        # final frame (post-terminal) with the resolved status
        top_rgb = _render_topdown(topdown_r, env.data, topdown_cam)
        fpv_rgb = _render_fpv(fpv_r, env.data)
        tail = _draw_frame(
            top_rgb, fpv_rgb, obs["depth"], cfg["env"]["depth_max_m"],
            traj, env.room, ep_idx, n_steps, env.n_collisions,
            float(obs["goal_vec"][0]), status,
        )
        # hold the last frame for 1 s
        for _ in range(10):
            writer.append_data(tail)
    finally:
        writer.close()
        topdown_r.close()
        fpv_r.close()

    return {
        "ep": ep_idx,
        "status": status,
        "n_steps": n_steps,
        "n_collisions": env.n_collisions,
        "final_goal_dist": float(obs["goal_vec"][0]),
        "path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override cfg.env.max_episode_steps for viz pacing")
    ap.add_argument("--out", default="results/demo",
                    help="output directory (one mp4 per episode)")
    ap.add_argument("--reel", action="store_true",
                    help="also write a single concatenated reel.mp4")
    ap.add_argument("--safety", action="store_true",
                    help="wrap the model policy with reactive safety override")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    policy = load_model_policy(args.ckpt, cfg, device, use_wa=True)
    if args.safety:
        policy = wrap_safety(policy, cfg)
        print("[safety] reactive override enabled "
              f"(near={cfg['safety']['near_m']}m, warn={cfg['safety']['warn_m']}m)")

    max_steps = args.max_steps or cfg["env"]["max_episode_steps"]
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = NavEnv(cfg["env"], seed=args.seed)
    summaries = []
    for ep_i in range(args.n_episodes):
        env.reset(seed=int(rng.integers(1 << 30)))
        ep_path = out_dir / f"ep{ep_i:03d}.mp4"
        summary = _run_one_episode(env, policy, cfg, ep_i, ep_path, max_steps)
        summaries.append(summary)
        print(f"[ep {ep_i}] status={summary['status']:9s}  steps={summary['n_steps']:3d}  "
              f"collisions={summary['n_collisions']:2d}  final_goal_dist={summary['final_goal_dist']:.2f} m  "
              f"→ {ep_path}")
    env.close()

    if args.reel:
        reel_path = out_dir / "reel.mp4"
        writer = imageio.get_writer(str(reel_path), fps=10, codec="libx264",
                                    quality=8, macro_block_size=1)
        for s in summaries:
            r = imageio.get_reader(s["path"])
            for f in r:
                writer.append_data(f)
            r.close()
        writer.close()
        print(f"reel → {reel_path}")

    n = len(summaries)
    n_ok = sum(1 for s in summaries if s["status"] == "success")
    n_col = sum(1 for s in summaries if s["status"] == "collision")
    n_to = sum(1 for s in summaries if s["status"] == "timeout")
    print(f"\nsummary: n={n}  success={n_ok}  collision={n_col}  timeout={n_to}  "
          f"SR={n_ok/n:.2f}")


if __name__ == "__main__":
    main()
