#!/usr/bin/env bash
# Experiments A (BEV channel ablation) and B (cross-distribution eval).
# Runs on the Pod. Assumes:
#   - /dev/shm/pib_nav_cache/*.npz present (1000 shards) OR regenerates
#   - configs/default.yaml has num_workers: 8
#   - scripts/train.py and scripts/eval.py work
#   - habitat conda env has torch + mujoco
#
# Experiment A protocol:
#   Train 3 leave-one-out channel ablations (no_occ, no_free, no_goal)
#   from scratch on PIB-Nav default data. Eval each at 4 seeds x +/-safety.
#   Use existing bev_ablation_pod/bev and /flat checkpoints as baselines.
#
# Experiment B protocol:
#   Generate a hard PIB-Nav test set (7-10m rooms, 6-10 obstacles,
#   smaller obstacles down to 0.25m).
#   Evaluate every trained variant on both (a) the original PIB-Nav
#   test split and (b) the hard split, paired 4-seed +/- safety.
#
# Total runtime estimate: ~2.5 h on RTX 4090, ~¥5.

set -euo pipefail
cd /root/data/bev-vawa-lite
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
HP=/root/miniconda3/envs/habitat/bin

RESULTS=/root/data/results/bev_A_B.csv
RUNS_ROOT=/root/data/runs/bev_A_B
DATA_DEFAULT=/dev/shm/pib_nav_cache
DATA_HARD=/dev/shm/pib_nav_hard_cache

mkdir -p /root/data/results /root/data/runs
rm -f "$RESULTS"

# ---------------------------------------------------------------
# Guard: ensure default PIB-Nav data is in tmpfs (regenerate if not)
# ---------------------------------------------------------------
if [ ! -d "$DATA_DEFAULT" ] || [ "$(ls "$DATA_DEFAULT"/*.npz 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "==== regenerating default PIB-Nav data ====" "$(date)"
    rm -rf "$DATA_DEFAULT"
    "$HP"/python scripts/generate_data.py --config configs/default.yaml \
        --out "$DATA_DEFAULT" --n-rooms 1000 --samples-per-room 50 2>&1 | tail -3
fi
echo "default PIB-Nav shards: $(ls "$DATA_DEFAULT"/*.npz | wc -l)"

# ---------------------------------------------------------------
# STEP A-1 -- train 3 channel ablations
# ---------------------------------------------------------------
train_variant () {
    local TAG="$1"; local CFG="$2"
    local OUT="$RUNS_ROOT/$TAG"
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "==== TRAIN $TAG Stage A ====" "$(date)"
    "$HP"/python scripts/train.py --config "$CFG" --stage a --data "$DATA_DEFAULT" --out "$OUT"
    echo "==== TRAIN $TAG Stage B ====" "$(date)"
    "$HP"/python scripts/train.py --config "$CFG" --stage b --data "$DATA_DEFAULT" --out "$OUT" --in-ckpt "$OUT/stage_a.pt"
    echo "==== TRAIN $TAG Stage C ====" "$(date)"
    "$HP"/python scripts/train.py --config "$CFG" --stage c --data "$DATA_DEFAULT" --out "$OUT" --in-ckpt "$OUT/stage_b.pt"
}

train_variant no_occ  configs/ablations/bev_no_occ.yaml
train_variant no_free configs/ablations/bev_no_free.yaml
train_variant no_goal configs/ablations/bev_no_goal.yaml

# ---------------------------------------------------------------
# STEP B-1 -- generate hard-distribution PIB-Nav data (test set only)
# ---------------------------------------------------------------
if [ ! -d "$DATA_HARD" ] || [ "$(ls "$DATA_HARD"/*.npz 2>/dev/null | wc -l)" -lt 50 ]; then
    echo "==== generating hard PIB-Nav data (200 rooms) ====" "$(date)"
    rm -rf "$DATA_HARD"
    "$HP"/python scripts/generate_data.py --config configs/ablations/pib_nav_hard.yaml \
        --out "$DATA_HARD" --n-rooms 200 --samples-per-room 50 2>&1 | tail -3
fi
echo "hard PIB-Nav shards: $(ls "$DATA_HARD"/*.npz | wc -l)"

# ---------------------------------------------------------------
# STEP 2 -- paired 4-seed eval for every variant on BOTH test distributions
# Variants list: checkpoint dir + config pairs. Use the Pod-trained
# bev_ablation_pod/bev and /flat if they exist; otherwise train fresh.
# ---------------------------------------------------------------
BEV_DIR=/root/data/runs/bev_ablation_pod/bev
FLAT_DIR=/root/data/runs/bev_ablation_pod/flat

[ -f "$BEV_DIR/stage_c.pt" ] || { echo "MISSING: $BEV_DIR/stage_c.pt — train it first"; exit 1; }
[ -f "$FLAT_DIR/stage_c.pt" ] || { echo "MISSING: $FLAT_DIR/stage_c.pt — train it first"; exit 1; }

VARIANTS=(
    "BEV-full:configs/default.yaml:$BEV_DIR"
    "Flat:configs/ablations/flat_encoder.yaml:$FLAT_DIR"
    "BEV-no_occ:configs/ablations/bev_no_occ.yaml:$RUNS_ROOT/no_occ"
    "BEV-no_free:configs/ablations/bev_no_free.yaml:$RUNS_ROOT/no_free"
    "BEV-no_goal:configs/ablations/bev_no_goal.yaml:$RUNS_ROOT/no_goal"
)

eval_variant_on_dist () {
    local TAG="$1"; local CFG="$2"; local CKPT="$3"; local DIST_LABEL="$4"
    for SEED in 12345 42 7 31337; do
        # no-safety
        "$HP"/python scripts/eval.py --config "$CFG" --policy bev_vawa \
            --ckpt "$CKPT" --n-episodes 100 --seed "$SEED" \
            --method-name "$TAG [$DIST_LABEL] seed=$SEED" --results "$RESULTS" 2>&1 | tail -1
        # +safety
        "$HP"/python scripts/eval.py --config "$CFG" --policy bev_vawa \
            --ckpt "$CKPT" --n-episodes 100 --seed "$SEED" \
            --method-name "$TAG [$DIST_LABEL] seed=$SEED +safety" --safety --results "$RESULTS" 2>&1 | tail -1
    done
}

echo "==== STEP 2: EVAL ALL VARIANTS x 2 distributions ====" "$(date)"

# (a) original / default distribution — sampled via NavEnv using
#     configs/default.yaml's env block.
for VAR in "${VARIANTS[@]}"; do
    TAG="${VAR%%:*}"; REST="${VAR#*:}"
    CFG="${REST%%:*}"; CKPT_DIR="${REST##*:}"
    eval_variant_on_dist "$TAG" "$CFG" "$CKPT_DIR/stage_c.pt" "default"
done

# (b) hard distribution — eval with the hard config so NavEnv samples
#     7-10m rooms with 6-10 obstacles. Model config stays same.
# NOTE: we use the hard config only for env sampling; the model
# architecture is determined by the checkpoint and doesn't change.
for VAR in "${VARIANTS[@]}"; do
    TAG="${VAR%%:*}"; REST="${VAR#*:}"
    # For cross-dist eval, pass pib_nav_hard.yaml as the config (env settings
    # differ, model settings inherit from default.yaml so are unchanged).
    CKPT_DIR="${REST##*:}"
    eval_variant_on_dist "$TAG" configs/ablations/pib_nav_hard.yaml \
        "$CKPT_DIR/stage_c.pt" "hard"
done

echo "==== ALL DONE ====" "$(date)"
cat "$RESULTS"
