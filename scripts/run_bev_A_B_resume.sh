#!/usr/bin/env bash
# Resume Experiment B hard-distribution eval for the 4 variants that the
# original run_bev_A_B_pod.sh crashed on (Flat / no_occ / no_free / no_goal).
#
# The original script passed configs/ablations/pib_nav_hard.yaml for EVERY
# variant in the hard loop, which works for BEV-full (whose ckpt is a pure
# BEV model matching default.yaml) but fails on:
#   - Flat:       ckpt has 1-channel input conv, BEV config has 3-channel
#   - BEV-no_occ: channels_enabled [0,1,1] must match the model built
#                 from the checkpoint
#   - BEV-no_free / BEV-no_goal: same reason as no_occ
#
# This resume pipeline uses per-variant composite configs of the form
# configs/ablations/<variant>_hard.yaml that inherit from the variant's
# default-distribution config and only override the env (room size,
# obstacles). The hard PIB-Nav shards already exist in
# /dev/shm/pib_nav_hard_cache from the original run; we do NOT regenerate.
#
# Appends to /root/data/results/bev_A_B.csv produced by the original run,
# so the final CSV covers all 80 eval cells.

set -euo pipefail
cd /root/data/bev-vawa-lite
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
HP=/root/miniconda3/envs/habitat/bin

RESULTS=/root/data/results/bev_A_B.csv
RUNS_ROOT=/root/data/runs/bev_A_B
BEV_DIR=/root/data/runs/bev_ablation_pod/bev
FLAT_DIR=/root/data/runs/bev_ablation_pod/flat

# Sanity checks -- do not start if the trained checkpoints are missing.
for CKPT in "$FLAT_DIR/stage_c.pt" "$RUNS_ROOT/no_occ/stage_c.pt" \
            "$RUNS_ROOT/no_free/stage_c.pt" "$RUNS_ROOT/no_goal/stage_c.pt"; do
    [ -f "$CKPT" ] || { echo "MISSING: $CKPT"; exit 1; }
done

echo "==== RESUME hard-dist eval for 4 variants ====" "$(date)"

eval_one () {
    local TAG="$1"; local CFG="$2"; local CKPT="$3"
    for SEED in 12345 42 7 31337; do
        "$HP"/python scripts/eval.py --config "$CFG" --policy bev_vawa \
            --ckpt "$CKPT" --n-episodes 100 --seed "$SEED" \
            --method-name "$TAG [hard] seed=$SEED" --results "$RESULTS" 2>&1 | tail -1
        "$HP"/python scripts/eval.py --config "$CFG" --policy bev_vawa \
            --ckpt "$CKPT" --n-episodes 100 --seed "$SEED" \
            --method-name "$TAG [hard] seed=$SEED +safety" --safety --results "$RESULTS" 2>&1 | tail -1
    done
}

eval_one "Flat"        configs/ablations/flat_encoder_hard.yaml "$FLAT_DIR/stage_c.pt"
eval_one "BEV-no_occ"  configs/ablations/bev_no_occ_hard.yaml   "$RUNS_ROOT/no_occ/stage_c.pt"
eval_one "BEV-no_free" configs/ablations/bev_no_free_hard.yaml  "$RUNS_ROOT/no_free/stage_c.pt"
eval_one "BEV-no_goal" configs/ablations/bev_no_goal_hard.yaml  "$RUNS_ROOT/no_goal/stage_c.pt"

echo "==== RESUME DONE ====" "$(date)"
echo "CSV now has $(wc -l < $RESULTS) lines (expected 81)"
