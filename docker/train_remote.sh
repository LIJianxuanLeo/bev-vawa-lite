#!/usr/bin/env bash
# Remote Habitat pipeline entrypoint (equivalent of `repro.sh` for the GPU box).
#
# Runs end-to-end inside the bev-vawa-lite:habitat container:
#   1. optional scene download
#   2. data generation from scenes → .npz shards
#   3. Stage A / B / C training
#   4. closed-loop evaluation on a held-out scene split
#
# Usage:
#   bash docker/train_remote.sh                # full run on the configured dataset
#   bash docker/train_remote.sh --tiny         # 1 scene, 1 epoch, 5 eval episodes
#   bash docker/train_remote.sh --dataset hssd        # use configs/habitat/hssd.yaml
#   bash docker/train_remote.sh --dataset procthor    # ProcTHOR-HAB
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

TINY=""
DATASET="hssd"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiny) TINY="--tiny"; shift;;
    --dataset) DATASET="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

CFG="configs/habitat/${DATASET}.yaml"
[[ -f "$CFG" ]] || { echo "config not found: $CFG"; exit 1; }

DATA_DIR="data/pib_nav_${DATASET}"
RUNS="runs/${DATASET}"
SCENES_GLOB="data/scene_datasets/${DATASET}/*.glb"
SCENES_TRAIN="data/scene_datasets/${DATASET}/train/*.glb"
SCENES_EVAL="data/scene_datasets/${DATASET}/val/*.glb"

# If the dataset isn't split into train/val, fall back to the full glob.
if ! compgen -G "$SCENES_TRAIN" > /dev/null; then SCENES_TRAIN="$SCENES_GLOB"; fi
if ! compgen -G "$SCENES_EVAL"  > /dev/null; then SCENES_EVAL="$SCENES_GLOB"; fi

echo "== [0/5] sanity =="
python -c "import torch, habitat_sim; print('cuda=', torch.cuda.is_available(), 'hsim=', habitat_sim.__version__)"
python -m pytest tests/ -q -k "not stage9 or not skip"  # MuJoCo tests still run here too

echo "== [1/5] data generation ($DATASET) =="
python scripts/generate_data_habitat.py \
    --config "$CFG" \
    --scenes $SCENES_TRAIN \
    --out    "$DATA_DIR" $TINY

echo "== [2/5] train Stage A =="
python scripts/train.py --config "$CFG" --stage a --data "$DATA_DIR" --out "$RUNS" $TINY
echo "== [3/5] train Stage B =="
python scripts/train.py --config "$CFG" --stage b --data "$DATA_DIR" --out "$RUNS" $TINY
echo "== [4/5] train Stage C =="
python scripts/train.py --config "$CFG" --stage c --data "$DATA_DIR" --out "$RUNS" $TINY

echo "== [5/5] evaluation =="
python scripts/eval_habitat.py \
    --config  "$CFG" \
    --scenes  $SCENES_EVAL \
    --policy  bev_vawa \
    --ckpt    "$RUNS/stage_c.pt" \
    --method-name "BEV-VAWA ($DATASET)" \
    $TINY

echo "Remote run ($DATASET) complete. See results/main_table_habitat.csv."
