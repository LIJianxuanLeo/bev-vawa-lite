"""Habitat-backed data generator (Gibson PointNav v2 track).

Usage on a remote CUDA + EGL host (see docker/README.md):

    python scripts/generate_data_habitat.py \
        --config      configs/habitat/gibson.yaml \
        --scene-dir   data/scene_datasets/gibson \
        --episode-dir data/datasets/pointnav/gibson/v2 \
        --split       train \
        --out         data/gibson_pointnav_v2_shards/train

Each scene produces ``scene_<stem>.npz`` shards in **schema v2** — they carry
``future_depth`` / ``future_goal`` (for the WA latent-dynamics loss) and
``cand_deadend`` (for the WA dead-end head) alongside the standard fields.
Downstream training (``scripts/train.py``) is unchanged.
"""
from __future__ import annotations
import argparse

from bev_vawa.utils import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/habitat/gibson.yaml")
    ap.add_argument("--dataset", default="gibson", choices=["gibson", "gibson_v2"],
                    help="Dataset source (only Gibson PointNav v2 is supported).")
    ap.add_argument("--scene-dir", required=True,
                    help="Dir containing Gibson .glb files, e.g. data/scene_datasets/gibson.")
    ap.add_argument("--episode-dir", required=True,
                    help="Dir with pointnav_gibson_v2 episodes, e.g. "
                         "data/datasets/pointnav/gibson/v2.")
    ap.add_argument("--split", default="train", help="One of train|val|test.")
    ap.add_argument("--out", default=None,
                    help="Shard output dir (defaults to cfg.data.out_dir).")
    ap.add_argument("--samples-per-episode", type=int, default=None)
    ap.add_argument("--max-episodes-per-scene", type=int, default=16,
                    help="How many episodes per scene to rollout.")
    ap.add_argument("--scene-limit", type=int, default=None,
                    help="Cap on the number of scenes to process.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device id for habitat-sim renderer.")
    ap.add_argument("--tiny", action="store_true",
                    help="Minimal sanity run: 2 scenes x 2 episodes x 2 samples.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    out_dir = args.out or cfg["data"]["out_dir"]

    samples_per_episode = args.samples_per_episode or int(
        cfg.get("gibson", {}).get("samples_per_episode", 8))
    max_ep = args.max_episodes_per_scene
    scene_limit = args.scene_limit
    if args.tiny:
        samples_per_episode = 2
        max_ep = 2
        scene_limit = 2

    from bev_vawa.data.rollout_habitat import generate_dataset_gibson
    n = generate_dataset_gibson(
        cfg,
        scene_dir=args.scene_dir,
        episode_dir=args.episode_dir,
        split=args.split,
        out_dir=out_dir,
        samples_per_episode=samples_per_episode,
        max_episodes_per_scene=max_ep,
        scene_limit=scene_limit,
        seed=args.seed,
        gpu_device_id=args.gpu,
    )
    print(f"wrote {n} Gibson v2 shards to {out_dir}")


if __name__ == "__main__":
    main()
