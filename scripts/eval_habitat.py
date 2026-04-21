"""Closed-loop evaluation on Habitat scenes (remote GPU track).

The model checkpoints are architecture-identical to the MuJoCo-trained ones,
so you can cross-evaluate: train on PIB-Nav (local) → eval on HSSD (remote),
or train on HSSD → eval on PIB-Nav.

Usage (inside docker/habitat image):

    python scripts/eval_habitat.py \
        --config configs/habitat/hssd.yaml \
        --scenes data/scene_datasets/hssd/*.glb \
        --policy bev_vawa --ckpt runs/remote/stage_c.pt \
        --n-episodes 100 \
        --method-name "BEV-VAWA (HSSD)"
"""
from __future__ import annotations
import argparse
import csv
import glob
import time
from pathlib import Path
import numpy as np
import torch

from bev_vawa.utils import load_config, get_device, set_seed
from bev_vawa.eval.metrics import summarize
from bev_vawa.models import FPV_BC, BEV_BC, BEV_VA, BEVVAWA


MODEL_MAP = {
    "fpv_bc": FPV_BC, "bev_bc": BEV_BC, "bev_va": BEV_VA,
    "bev_vawa": BEVVAWA,
}


def _expand(patterns):
    out = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches and Path(p).exists():
            matches = [p]
        out.extend(matches)
    return sorted(set(out))


def _build_policy(args, cfg, device):
    if args.policy == "straight":
        p = lambda obs, cfg: (cfg["env"]["max_lin_vel"], 0.0)
    elif args.policy in MODEL_MAP:
        assert args.ckpt, "--ckpt required for model policies"
        cls = MODEL_MAP[args.policy]
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = cls(cfg).to(device)
        model.load_state_dict(state["model"], strict=False)
        from bev_vawa.eval.policies import make_model_policy
        p = make_model_policy(model, device, cfg, use_wa=(args.policy == "bev_vawa"))
    else:
        raise ValueError(args.policy)

    if getattr(args, "safety", False):
        # Reactive obstacle-avoidance wrapper. Uses forward + side-sector
        # depth clearances to override (v, omega) near collisions; never
        # raises v. Matches the PIB-Nav eval path bit-for-bit so Habitat
        # numbers are directly comparable to the main table.
        from bev_vawa.eval.policies import wrap_safety
        p = wrap_safety(p, cfg)
    return p


def _run_one_episode(env, policy, cfg) -> dict:
    obs = env._get_obs()
    prev_xy = np.asarray([obs["pose"][0], obs["pose"][1]]).copy()
    agent_path = 0.0
    latencies = []
    steps = 0
    success = False
    while True:
        t0 = time.time()
        v, w = policy(obs, cfg)
        latencies.append((time.time() - t0) * 1000.0)
        step = env.step((v, w))
        obs = step.obs
        xy = np.asarray([obs["pose"][0], obs["pose"][1]])
        agent_path += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy.copy()
        steps += 1
        if step.info["reached"]:
            success = True
            break
        if step.done:
            break
    return {
        "success": bool(success),
        "shortest": float(env.shortest_distance),
        "agent": float(agent_path),
        "collisions": int(env.n_collisions),
        "steps": int(steps),
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/habitat/default.yaml")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--policy", required=True,
                    choices=["straight"] + list(MODEL_MAP.keys()))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--results", default="results/main_table_habitat.csv")
    ap.add_argument("--method-name", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--safety", action="store_true",
                    help="wrap the policy with the reactive obstacle-avoidance "
                         "override (same wrapper as the PIB-Nav eval).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    n_eps = 5 if args.tiny else args.n_episodes

    scenes = _expand(args.scenes)
    if not scenes:
        raise SystemExit(f"no scenes matched {args.scenes!r}")

    # defer habitat import
    from bev_vawa.envs.habitat_env import HabitatNavEnv

    policy = _build_policy(args, cfg, device)
    rng = np.random.default_rng(args.seed)
    eps = []
    for ep_i in range(n_eps):
        scene = scenes[ep_i % len(scenes)]
        env = HabitatNavEnv(cfg["env"], scene_glb=scene,
                            seed=int(rng.integers(1 << 30)),
                            gpu_device_id=args.gpu)
        try:
            env.reset(seed=int(rng.integers(1 << 30)))
            eps.append(_run_one_episode(env, policy, cfg))
        finally:
            env.close()
    summary = summarize(eps)
    print("summary:", {k: v for k, v in summary.items() if k != "per_episode"})

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["method", "SR", "SPL", "CollisionRate", "PathLenRatio", "LatencyMs"]
    rows = []
    if out.exists():
        with open(out, "r") as f:
            rows = list(csv.DictReader(f))
    # auto-suffix with "+safety" so a single results CSV can hold both
    # with- and without-safety rows for the same policy/seed without
    # silently overwriting.
    base_label = args.method_name or args.policy
    label = f"{base_label} +safety" if args.safety and "+safety" not in base_label else base_label
    row = {"method": label, "SR": summary["SR"], "SPL": summary["SPL"],
           "CollisionRate": summary["CollisionRate"],
           "PathLenRatio": summary["PathLenRatio"], "LatencyMs": summary["LatencyMs"]}
    rows = [r for r in rows if r.get("method") != label]
    rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
