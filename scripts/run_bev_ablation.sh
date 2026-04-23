#!/usr/bin/env bash
# BEV ablation runner — trains and evaluates the "flat" encoder variant
# (no geometric lift) at the same 4 seeds used by the main BEV-VAWA
# cross-seed analysis. The BEV variant's numbers come from the existing
# main_table.csv, so only the flat variant needs to be trained here.
#
# PIB-Nav is a MuJoCo benchmark that runs fine on CPU/MPS; this script
# is designed to run on the Apple M4 laptop (zero remote GPU cost).
# Total wall-clock: ~3 h on M4 (4 seeds x A-B-C training + eval).
#
# Usage:
#   bash scripts/run_bev_ablation.sh
#
# Outputs:
#   runs/flat_encoder/{seed_12345,seed_42,seed_7,seed_31337}/stage_{a,b,c}.pt
#   results/bev_ablation.csv   (4 seeds x 2 conditions with/without safety)

set -euxo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

CFG=configs/ablations/flat_encoder.yaml
DATA=data/pib_nav
BASE_OUT=runs/flat_encoder
RESULTS=results/bev_ablation.csv

rm -f "$RESULTS"

for SEED in 12345 42 7 31337; do
    RUN_DIR="$BASE_OUT/seed_$SEED"
    mkdir -p "$RUN_DIR"

    echo "==== flat-encoder seed=$SEED: Stage A ===="
    python scripts/train.py --config "$CFG" --stage a \
        --data "$DATA" --out "$RUN_DIR"

    echo "==== flat-encoder seed=$SEED: Stage B ===="
    python scripts/train.py --config "$CFG" --stage b \
        --data "$DATA" --out "$RUN_DIR" \
        --in-ckpt "$RUN_DIR/stage_a.pt"

    echo "==== flat-encoder seed=$SEED: Stage C ===="
    python scripts/train.py --config "$CFG" --stage c \
        --data "$DATA" --out "$RUN_DIR" \
        --in-ckpt "$RUN_DIR/stage_b.pt"

    echo "==== flat-encoder seed=$SEED: eval (no safety) ===="
    python scripts/eval.py --config "$CFG" \
        --policy bev_vawa --ckpt "$RUN_DIR/stage_c.pt" \
        --n-episodes 100 --seed $SEED \
        --method-name "Flat Encoder seed=$SEED" \
        --results "$RESULTS"

    echo "==== flat-encoder seed=$SEED: eval (+ safety) ===="
    # Explicit +safety suffix because scripts/eval.py (the PIB-Nav
    # evaluator) does NOT auto-append the flag to --method-name, unlike
    # scripts/eval_habitat.py. Without a distinct name, the +safety row
    # would overwrite the no-safety row in the CSV (same method key).
    python scripts/eval.py --config "$CFG" \
        --policy bev_vawa --ckpt "$RUN_DIR/stage_c.pt" \
        --n-episodes 100 --seed $SEED \
        --method-name "Flat Encoder seed=$SEED +safety" --safety \
        --results "$RESULTS"
done

echo "==== DONE ===="
cat "$RESULTS"
