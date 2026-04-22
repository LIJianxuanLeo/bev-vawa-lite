#!/usr/bin/env bash
# Single DAGger iteration on Gibson:
#   1. aggregate policy rollout samples with pathfinder-relabelled experts
#   2. mix DAGger shards + original teleport shards via symlinks
#   3. re-train Stage A / B / C on the mixed data (10/10/5 epochs)
#   4. paired 4-seed eval (no-safety + safety) on Gibson val
#
# Prerequisites in the Pod:
#   /root/data/runs/gibson/stage_c.pt               # baseline checkpoint
#   /root/data/gibson_pointnav_v2_shards/train/*.npz  # teleport shards
#   /root/data/scene_datasets/gibson/*.glb          # scenes
#   /root/data/datasets/pointnav/gibson/v2/...      # episodes
#   conda env `habitat` with hsim 0.3.3 + project installed
#
# Usage:  bash scripts/run_dagger_iteration.sh [iter_tag] [beta]
#   iter_tag defaults to "iter1" — pass e.g. "iter2" to chain rounds.
#   beta defaults to 0.0 — pure policy rollout (original DAGger-1). Use a
#        schedule like 0.8 → 0.4 → 0.0 across three chained iterations to
#        smoothly transition from teleport-expert-like to off-policy DAGger.
#
# Environment overrides:
#   CKPT_IN        — source checkpoint for the rollout policy (default:
#                    /root/data/runs/gibson/stage_c.pt). Pass the previous
#                    iteration's Stage-C ckpt when chaining.
#
# Total runtime: ~90 min (mostly the aggregation rollout) → ~¥2.8.

set -euxo pipefail
cd /root/data/bev-vawa-lite
export PYTHONPATH=$PWD
HP=/root/miniconda3/envs/habitat/bin

ITER="${1:-iter1}"
BETA="${2:-0.0}"
CKPT_IN="${CKPT_IN:-/root/data/runs/gibson/stage_c.pt}"

DAGGER_DIR="/root/data/gibson_dagger_shards/${ITER}"
MIX_DIR="/root/data/gibson_mix_shards/${ITER}"
RUN_DIR="/root/data/runs/gibson_dagger_${ITER}"

# ------------------------------------------------------------------
# 1. Aggregate closed-loop samples, pathfinder-relabelled
# ------------------------------------------------------------------
echo "==== STEP 1: DAGger aggregation (${ITER}, beta=${BETA}) ==== $(date)"
$HP/python scripts/dagger_aggregate_habitat.py \
    --config configs/habitat/gibson.yaml \
    --ckpt "$CKPT_IN" \
    --beta "$BETA" \
    --scene-dir /root/data/scene_datasets/gibson \
    --episode-dir /root/data/datasets/pointnav/gibson/v2 \
    --split train \
    --out "$DAGGER_DIR" \
    --max-episodes-per-scene 4 \
    --max-steps-per-episode 150

# ------------------------------------------------------------------
# 2. Union the teleport shards with DAGger shards (via symlink)
# ------------------------------------------------------------------
echo "==== STEP 2: Mix shards ==== $(date)"
mkdir -p "$MIX_DIR"
rm -f "$MIX_DIR"/*.npz
# Preserve scene_*.npz naming so NavShardDataset's list_shards globber
# (`scene_*.npz` and `dagger_*.npz`) catches everything. Teleport shards
# are already named scene_<scene>.npz; we link them unchanged. DAGger
# shards are named dagger_<scene>.npz in the aggregator; also matched.
for f in /root/data/gibson_pointnav_v2_shards/train/*.npz; do
    ln -sf "$f" "$MIX_DIR/$(basename "$f")"
done
for f in "$DAGGER_DIR"/*.npz; do
    ln -sf "$f" "$MIX_DIR/$(basename "$f")"
done
echo "mix shards: $(ls $MIX_DIR/*.npz | wc -l)"

# ------------------------------------------------------------------
# 3. Retrain all three stages on the mixed data
# ------------------------------------------------------------------
echo "==== STEP 3: Retrain A/B/C ==== $(date)"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

$HP/python scripts/train.py --config configs/habitat/gibson.yaml \
    --stage a --data "$MIX_DIR" --out "$RUN_DIR" --epochs 10

$HP/python scripts/train.py --config configs/habitat/gibson.yaml \
    --stage b --data "$MIX_DIR" --out "$RUN_DIR" \
    --in-ckpt "$RUN_DIR/stage_a.pt" --epochs 10

$HP/python scripts/train.py --config configs/habitat/gibson.yaml \
    --stage c --data "$MIX_DIR" --out "$RUN_DIR" \
    --in-ckpt "$RUN_DIR/stage_b.pt" --epochs 5

# ------------------------------------------------------------------
# 4. Paired 4-seed eval on Gibson val
# ------------------------------------------------------------------
echo "==== STEP 4: Paired 4-seed eval ==== $(date)"
VAL_SCENES=()
for f in /root/data/datasets/pointnav/gibson/v2/val/content/*.json.gz; do
    name=$(basename "$f" .json.gz)
    glb="/root/data/scene_datasets/gibson/${name}.glb"
    [ -f "$glb" ] && VAL_SCENES+=("$glb")
done

RESULTS_CSV="$RUN_DIR/dagger_eval.csv"
rm -f "$RESULTS_CSV"

for SEED in 12345 42 7 31337; do
    $HP/python scripts/eval_habitat.py \
        --config configs/habitat/gibson.yaml \
        --scenes "${VAL_SCENES[@]}" \
        --policy bev_vawa --ckpt "$RUN_DIR/stage_c.pt" \
        --n-episodes 100 --seed $SEED \
        --method-name "BEV-VAWA DAGger-${ITER} seed=${SEED}" \
        --results "$RESULTS_CSV"
done
for SEED in 12345 42 7 31337; do
    $HP/python scripts/eval_habitat.py \
        --config configs/habitat/gibson.yaml \
        --scenes "${VAL_SCENES[@]}" \
        --policy bev_vawa --ckpt "$RUN_DIR/stage_c.pt" \
        --n-episodes 100 --seed $SEED \
        --method-name "BEV-VAWA DAGger-${ITER} seed=${SEED}" --safety \
        --results "$RESULTS_CSV"
done

echo "==== DAGger ${ITER} DONE ==== $(date)"
echo "--- results CSV ---"
cat "$RESULTS_CSV"
