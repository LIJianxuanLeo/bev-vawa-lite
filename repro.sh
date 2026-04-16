#!/usr/bin/env bash
# Reproducibility entrypoint. Default is --tiny which runs the full pipeline
# end-to-end on an Apple M4 in under a few minutes for reviewer sanity.
#
# Usage:
#   bash repro.sh                 # tiny repro (default)
#   bash repro.sh --full          # full data + train + eval (hours)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

MODE="${1:-tiny}"
case "$MODE" in
  --tiny|tiny)
    TINY="--tiny"
    DATA="data/pib_nav_tiny"
    RUNS="runs/tiny"
    ;;
  --full|full)
    TINY=""
    DATA="data/pib_nav"
    RUNS="runs/full"
    ;;
  *)
    echo "usage: bash repro.sh [--tiny|--full]"; exit 1
    ;;
esac

echo "== [0/6] pytest =="
python3 -m pytest tests/ -q

echo "== [1/6] data generation =="
python3 scripts/generate_data.py --out "$DATA" $TINY

echo "== [2/6] train stage A =="
python3 scripts/train.py --stage a --data "$DATA" --out "$RUNS" $TINY

echo "== [3/6] train stage B =="
python3 scripts/train.py --stage b --data "$DATA" --out "$RUNS" $TINY

echo "== [4/6] train stage C =="
python3 scripts/train.py --stage c --data "$DATA" --out "$RUNS" $TINY

echo "== [5/6] eval (A* oracle + full model) =="
python3 scripts/eval.py --policy astar --method-name "A* Upper Bound" $TINY
python3 scripts/eval.py --policy bev_vawa --ckpt "$RUNS/stage_c.pt" \
    --method-name "BEV-VAWA (Ours)" $TINY

echo "== [6/6] figures =="
python3 scripts/make_figures.py --dummy

echo "Repro $MODE complete. See results/ for CSVs and figures."
