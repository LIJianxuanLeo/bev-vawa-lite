# BEV-VAWA-Lite

Lightweight BEV-native **Vision-Action / World-Action** navigation for
depth-only static indoor PointGoal. Runs end-to-end on a consumer laptop
(Apple Silicon, MPS backend). Designed as a one-month research project
target.

## Architecture at a Glance

```
Depth + Goal
   → Shared BEV Encoder (CNN + Lift-to-BEV + LSTM)  → latent z
   → VA head: K=5 anchor logits + small offsets
   → WA head: H=3 latent rollout → risk / progress / uncertainty
   → Fusion  Q = α s + β p − γ r − δ u
   → Pure-pursuit controller → (v, ω)
   → MuJoCo differential-drive robot
```

All training and evaluation happen in **PIB-Nav**, a procedural MuJoCo
indoor benchmark we build on-the-fly (rectangular rooms, random obstacles,
ground-truth occupancy + A* expert).

## Install

```bash
# Python 3.10+ recommended. Torch >= 2.4 to dodge the PyTorch-MPS
# non-contiguous in-place-op bug on older macOS.
pip install -e .
```

Dependencies: `torch>=2.4`, `mujoco>=3.1`, `numpy`, `pyyaml`, `tqdm`,
`imageio`, `matplotlib`, `pytest`.

## Quickstart (tiny, <5 minutes on M4)

```bash
bash repro.sh --tiny
```

This runs the full pytest suite, generates a mini dataset, trains Stage A/B/C
each for 1 epoch × 3 batches, evaluates the A\* oracle and full model on 5
episodes, and writes figures to `results/`.

## Full Pipeline

```bash
# 1. Offline data (~200 procedural rooms, ~12k samples, ~2 GB on disk)
python scripts/generate_data.py

# 2. Train VA -> WA -> joint
python scripts/train.py --stage a
python scripts/train.py --stage b
python scripts/train.py --stage c

# 3. Baselines (paper §9.1)
python scripts/train.py --stage fpv_bc
python scripts/train.py --stage bev_bc
python scripts/train.py --stage bev_va

# 4. Closed-loop evaluation
python scripts/eval.py --policy astar    --method-name "A* Upper Bound"
python scripts/eval.py --policy fpv_bc   --ckpt runs/default/fpv_bc.pt --method-name FPV-BC
python scripts/eval.py --policy bev_bc   --ckpt runs/default/bev_bc.pt --method-name BEV-BC
python scripts/eval.py --policy bev_va   --ckpt runs/default/bev_va.pt --method-name BEV-VA
python scripts/eval.py --policy bev_vawa --ckpt runs/default/stage_c.pt --method-name "BEV-VAWA (Ours)"

# 5. Figures + paper draft
python scripts/make_figures.py
```

## Repository Layout

```
bev_vawa/
  envs/           # procedural MuJoCo env + occupancy + A*
  data/           # expert labels + offline rollout + torch Dataset
  models/         # BEV encoder, VA/WA heads, fusion, baselines
  train/          # Stage A/B/C trainers + baseline trainer
  control/        # pure-pursuit
  eval/           # closed-loop runner + metrics + policies
configs/          # default.yaml + ablations/
scripts/          # generate_data / train / eval / make_figures
tests/            # one pytest file per stage gate — always green
paper/paper.md    # paper draft skeleton
results/          # main_table.csv / ablation_table.csv / figures
```

## Stage Gates (reusable)

Every stage in `tests/` is a regression gate re-run by `pytest tests/`:

| Stage | Gate file                      | What it checks                           |
|-------|--------------------------------|------------------------------------------|
| 0     | `test_stage0_env.py`           | deps import, MPS conv, MuJoCo depth      |
| 1     | `test_stage1_env.py`           | procedural rooms + A* + env step bounds  |
| 2     | `test_stage2_data.py`          | offline dataset shapes + label sanity    |
| 3     | `test_stage3_va.py`            | model size, MPS latency, Stage-A smoke   |
| 4     | `test_stage4_wa.py`            | fusion math, Stage-B/C smoke             |
| 5     | `test_stage5_loop.py`          | controller + closed-loop + SPL           |
| 6     | `test_stage6_baselines.py`     | baselines train, ablation configs load   |
| 7     | `test_stage7_artifacts.py`     | paper sections + figure pipeline         |

## Notes on Apple Silicon

* Training uses `torch.backends.mps`. `get_device()` picks MPS automatically.
* We default to **LSTM** (not GRU) because PyTorch-MPS GRU has known slowness.
* MuJoCo depth rendering prints an `ARB_clip_control unavailable` warning on
  macOS; this is a known macOS GL limitation and does not affect training,
  only depth precision near the far plane.

## License

Research use only.
