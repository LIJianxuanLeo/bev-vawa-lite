#!/usr/bin/env bash
# Remote Habitat pipeline entrypoint (equivalent of `repro.sh` for a GPU box).
#
# Runs end-to-end inside the bev-vawa-lite:habitat container:
#   1. sanity check (torch.cuda + habitat_sim import)
#   2. data generation from Gibson PointNav v2 scenes → schema-v2 .npz shards
#   3. Stage A / B / C training
#   4. closed-loop evaluation on the val split
#
# Usage:
#   bash docker/train_remote.sh          # full run
#   bash docker/train_remote.sh --tiny   # 1 scene, 1 epoch, 5 eval episodes
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

TINY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiny) TINY="--tiny"; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

CFG="configs/habitat/gibson.yaml"
DATA_DIR="data/gibson_pointnav_v2_shards"
RUNS="runs/gibson"
SCENE_DIR="data/scene_datasets/gibson"
EPISODE_DIR="data/datasets/pointnav/gibson/v2"
METHOD_NAME="BEV-VAWA (Gibson)"

[[ -f "$CFG" ]] || { echo "config not found: $CFG"; exit 1; }

echo "== [0/5] sanity =="
python -c "import torch, habitat_sim; print('cuda=', torch.cuda.is_available(), 'hsim=', habitat_sim.__version__)"
python -m pytest tests/ -q -k "not stage9 or not skip"

echo "== [1/5] data generation (Gibson PointNav v2) =="
python scripts/generate_data_habitat.py \
    --config "$CFG" \
    --dataset gibson_v2 \
    --scene-dir "$SCENE_DIR" \
    --episode-dir "$EPISODE_DIR" \
    --split train \
    --out "$DATA_DIR/train" $TINY
python scripts/generate_data_habitat.py \
    --config "$CFG" \
    --dataset gibson_v2 \
    --scene-dir "$SCENE_DIR" \
    --episode-dir "$EPISODE_DIR" \
    --split val \
    --out "$DATA_DIR/val" $TINY

echo "== [2/5] train Stage A =="
python scripts/train.py --config "$CFG" --stage a --data "$DATA_DIR/train" --out "$RUNS" $TINY
echo "== [3/5] train Stage B =="
python scripts/train.py --config "$CFG" --stage b --data "$DATA_DIR/train" --out "$RUNS" $TINY
echo "== [4/5] train Stage C =="
python scripts/train.py --config "$CFG" --stage c --data "$DATA_DIR/train" --out "$RUNS" $TINY

echo "== [5/5] evaluation =="
python scripts/eval_habitat.py \
    --config  "$CFG" \
    --scenes  "$SCENE_DIR/*.glb" \
    --policy  bev_vawa \
    --ckpt    "$RUNS/stage_c.pt" \
    --method-name "$METHOD_NAME" \
    $TINY

echo "Remote run (Gibson) complete. See results/main_table_habitat.csv."
