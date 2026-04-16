"""Train one stage (a/b/c) or a baseline (fpv_bc/bev_bc/bev_va)."""
from __future__ import annotations
import argparse
from pathlib import Path

from bev_vawa.utils import load_config
from bev_vawa.train import train_stage_a, train_stage_b, train_stage_c, train_baseline
from bev_vawa.models import FPV_BC, BEV_BC, BEV_VA


BASELINE_MAP = {"fpv_bc": FPV_BC, "bev_bc": BEV_BC, "bev_va": BEV_VA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default=None, help="dataset dir (defaults to cfg.data.out_dir)")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--stage", choices=["a", "b", "c"] + list(BASELINE_MAP.keys()), required=True)
    ap.add_argument("--in-ckpt", default=None, help="checkpoint to resume from (stage b/c)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="1 epoch x 3 batches")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data or cfg["data"]["out_dir"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.tiny else args.epochs
    max_batches = 3 if args.tiny else None

    if args.stage == "a":
        res = train_stage_a(cfg, data_dir, str(out_dir / "stage_a.pt"), epochs=epochs, max_batches=max_batches)
    elif args.stage == "b":
        in_ck = args.in_ckpt or str(out_dir / "stage_a.pt")
        res = train_stage_b(cfg, data_dir, in_ck, str(out_dir / "stage_b.pt"), epochs=epochs, max_batches=max_batches)
    elif args.stage == "c":
        in_ck = args.in_ckpt or str(out_dir / "stage_b.pt")
        res = train_stage_c(cfg, data_dir, in_ck, str(out_dir / "stage_c.pt"), epochs=epochs, max_batches=max_batches)
    else:
        cls = BASELINE_MAP[args.stage]
        res = train_baseline(cls, cfg, data_dir, str(out_dir / f"{args.stage}.pt"),
                             epochs=epochs, max_batches=max_batches)
    print("final:", res)


if __name__ == "__main__":
    main()
