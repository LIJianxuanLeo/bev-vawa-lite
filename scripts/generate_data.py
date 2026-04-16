"""Offline dataset generator: fills data/pib_nav with .npz shards."""
from __future__ import annotations
import argparse
from pathlib import Path

from bev_vawa.utils import load_config, set_seed
from bev_vawa.data import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="override data.out_dir")
    ap.add_argument("--n-rooms", type=int, default=None)
    ap.add_argument("--samples-per-room", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="5 rooms x 8 samples (repro smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    out_dir = args.out or cfg["data"]["out_dir"]
    n_rooms = args.n_rooms or cfg["data"]["n_rooms"]
    spr = args.samples_per_room or cfg["data"]["samples_per_room"]
    if args.tiny:
        n_rooms, spr = 5, 8

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    generate_dataset(cfg, out_dir, n_rooms=n_rooms, samples_per_room=spr, seed=seed, verbose=True)


if __name__ == "__main__":
    main()
