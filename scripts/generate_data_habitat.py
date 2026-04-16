"""Habitat-backed data generator (remote GPU track).

Usage on a remote CUDA + EGL host (see docker/README.md):

    python scripts/generate_data_habitat.py \
        --config configs/habitat/hssd.yaml \
        --scenes data/scene_datasets/hssd/*.glb \
        --out   data/pib_nav_habitat \
        --pairs-per-scene 16 --samples-per-pair 8

Each scene produces a single ``scene_<stem>.npz`` shard in the same format as
``scripts/generate_data.py`` (the MuJoCo version), so downstream training
(``scripts/train.py``) is unchanged.
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

from bev_vawa.utils import load_config, set_seed


def _expand_scenes(patterns):
    out = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches and Path(p).exists():
            matches = [p]
        out.extend(matches)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/habitat/default.yaml")
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Scene file(s) or glob(s). E.g. 'data/hssd/*.glb'.")
    ap.add_argument("--out", default=None, help="shard output dir (defaults to cfg.data.out_dir)")
    ap.add_argument("--pairs-per-scene", type=int, default=None)
    ap.add_argument("--samples-per-pair", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device id for habitat-sim renderer")
    ap.add_argument("--tiny", action="store_true",
                    help="1 scene x 2 pairs x 4 samples — for docker sanity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    out_dir = args.out or cfg["data"]["out_dir"]

    # default knobs from cfg, overrideable from CLI
    hab = cfg.get("habitat", {})
    n_pairs = args.pairs_per_scene or int(hab.get("pairs_per_scene", 16))
    n_per_pair = args.samples_per_pair or int(hab.get("samples_per_pair", 8))

    scenes = _expand_scenes(args.scenes)
    if args.tiny:
        scenes = scenes[:1]
        n_pairs = 2
        n_per_pair = 4

    if not scenes:
        raise SystemExit(f"no scenes matched {args.scenes!r}")

    # defer import to here: habitat-sim is only required inside the docker image
    from bev_vawa.data.rollout_habitat import generate_dataset_habitat

    n = generate_dataset_habitat(
        cfg, scenes, out_dir,
        n_pairs_per_scene=n_pairs,
        samples_per_pair=n_per_pair,
        seed=args.seed,
        gpu_device_id=args.gpu,
    )
    print(f"wrote {n} shards to {out_dir}")


if __name__ == "__main__":
    main()
