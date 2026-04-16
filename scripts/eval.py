"""Closed-loop evaluation of a policy (full, baseline, or A* oracle) on PIB-Nav.

Writes a per-method row into ``results/main_table.csv`` (appending / updating).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from bev_vawa.utils import load_config, get_device
from bev_vawa.envs import NavEnv
from bev_vawa.eval.closed_loop import run_episode
from bev_vawa.eval.metrics import summarize
from bev_vawa.eval.policies import make_goal_policy, make_astar_policy, load_model_policy
from bev_vawa.models import FPV_BC, BEV_BC, BEV_VA, BEVVAWA


MODEL_MAP = {
    "fpv_bc": FPV_BC, "bev_bc": BEV_BC, "bev_va": BEV_VA, "bev_vawa": BEVVAWA,
}


def _build_policy(args, cfg, device):
    if args.policy == "astar":
        # a per-episode factory: we rebuild each reset because A* is scene-specific
        return "astar"
    if args.policy == "goal":
        return make_goal_policy()
    if args.policy in MODEL_MAP:
        assert args.ckpt, "--ckpt is required for model policies"
        cls = MODEL_MAP[args.policy]
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = cls(cfg).to(device)
        model.load_state_dict(state["model"], strict=False)
        from bev_vawa.eval.policies import make_model_policy
        return make_model_policy(model, device, cfg, use_wa=(args.policy == "bev_vawa"))
    raise ValueError(args.policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", required=True,
                    choices=["goal", "astar", "fpv_bc", "bev_bc", "bev_va", "bev_vawa"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--results", default="results/main_table.csv")
    ap.add_argument("--method-name", default=None, help="row label in the CSV")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    n_eps = 5 if args.tiny else (args.n_episodes or cfg["eval"]["n_episodes"])

    policy_obj = _build_policy(args, cfg, device)
    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(1 << 30)) for _ in range(n_eps)]
    env = NavEnv(cfg["env"], seed=args.seed)
    eps = []
    for s in seeds:
        env.reset(seed=s)
        if policy_obj == "astar":
            p = make_astar_policy(env.room, cfg)
        else:
            p = policy_obj
        eps.append(run_episode(env, p, cfg))
    env.close()
    summary = summarize(eps)
    print("summary:", {k: v for k, v in summary.items() if k != "per_episode"})

    # append to CSV
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["method", "SR", "SPL", "CollisionRate", "PathLenRatio", "LatencyMs"]
    rows = []
    if out.exists():
        with open(out, "r") as f:
            rows = list(csv.DictReader(f))
    label = args.method_name or args.policy
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
