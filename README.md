# BEV-VAWA-Lite

> **BEV-native Vision-Action / World-Action navigation for depth-only indoor PointGoal.**
> Runs end-to-end on a consumer laptop (Apple Silicon, PyTorch MPS backend).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Shared BEV Encoder](#1-shared-bev-encoder)
   - [Vision-Action (VA) Branch](#2-vision-action-va-branch)
   - [World-Action (WA) Branch](#3-world-action-wa-branch)
   - [Fusion & Waypoint Selection](#4-fusion--waypoint-selection)
   - [Pure-Pursuit Controller](#5-pure-pursuit-controller)
3. [PIB-Nav Benchmark](#pib-nav-benchmark)
4. [Data Pipeline](#data-pipeline)
5. [Training](#training)
   - [Two Tracks: Local (MuJoCo) vs Remote (Habitat)](#two-tracks-local-mujoco-vs-remote-habitat)
   - [Local Track — Stage A/B/C on PIB-Nav (Apple M4)](#local-track--stage-abc-on-pib-nav-apple-m4)
     - [Stage A — VA Imitation](#stage-a--va-imitation)
     - [Stage B — WA Future Prediction](#stage-b--wa-future-prediction)
     - [Stage C — Joint Fine-tuning](#stage-c--joint-fine-tuning)
   - [Remote Track — Habitat Scenes on a Linux GPU Box](#remote-track--habitat-scenes-on-a-linux-gpu-box)
6. [Baselines & Ablations](#baselines--ablations)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Install](#install)
9. [Quickstart](#quickstart-tiny-5-minutes-on-m4)
10. [Full Pipeline](#full-pipeline)
11. [Configuration System](#configuration-system)
12. [Repository Layout](#repository-layout)
13. [Stage Gates](#stage-gates)
14. [Engineering Decisions](#engineering-decisions)
15. [License](#license)

---

## Overview

**BEV-VAWA-Lite** is a lightweight navigation agent that addresses the indoor PointGoal task: navigate a depth-camera robot from a random start pose to a goal position in an unknown room, avoiding obstacles. The agent is designed to run on commodity hardware (Apple M4) with no GPU — training completes in roughly 10–20 hours total on MPS.

The key architectural idea is to maintain a **Bird's-Eye-View (BEV) latent state** that is shared between two prediction branches:

- **Vision-Action (VA)** branch: directly scores K candidate waypoints by imitating an A\* expert.  
- **World-Action (WA)** branch: rolls out each candidate in latent space and predicts collision risk, forward progress, and uncertainty — effectively a lightweight *world model*.

A simple linear fusion rule then selects the best waypoint for each timestep. A pure-pursuit controller converts the waypoint into differential-drive commands (v, ω).

The benchmark — **PIB-Nav** (Procedural Indoor Benchmark for Navigation) — is built procedurally in MuJoCo. It requires no external scene assets and runs headlessly on macOS without EGL, making it a fully self-contained research testbed.

---

## Architecture

```
Depth (128×128) + Goal vec (2D)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  Shared BEV Encoder                     │
│   CNN (32→64→96)                                        │
│     └─ Lift-to-BEV (AdaptiveAvgPool → 1×1 Conv)        │
│         └─ BEV pool → FC → latent_dim=128               │
│   LSTM-128 (goal-conditioned recurrent update)          │
│                           → z_t ∈ ℝ¹²⁸                 │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐     ┌──────────────────────────┐
│  VA Head      │     │  WA Head                 │
│               │     │  anchor_embed(wp) → E    │
│  MLP(z) →     │     │  LSTMCell ×H rollout     │
│  K logits     │     │  └─ risk   (BCE ensemble)│
│  K offsets    │     │  └─ progress (MSE)       │
│               │     │  └─ uncertainty (var)    │
└───────┬───────┘     └──────────┬───────────────┘
        │                        │
        └───────────┬────────────┘
                    ▼
         Fusion: Q_k = α·s_k + β·p_k − γ·r_k − δ·u_k
                    │
                    ▼
           argmax_k Q_k  →  selected waypoint (robot frame)
                    │
                    ▼
        Pure-Pursuit Controller  →  (v, ω)
                    │
                    ▼
         MuJoCo differential-drive robot
```

### 1. Shared BEV Encoder

**File:** `bev_vawa/models/bev_encoder.py`

Takes a single-channel depth frame `(B, 1, 128, 128)` and a 2D goal vector `(B, 2)`.

| Sub-module | Detail |
|---|---|
| **CNN** | Three stride-2 convolutions: 1→32→64→96 channels; field of view 5×5 then 3×3. Spatial size shrinks to 16×16. |
| **Lift-to-BEV** | `AdaptiveAvgPool2d` to 64×64 followed by a 1×1 conv projecting to 64 channels. Approximates a top-down occupancy feature without full frustum unprojection. |
| **BEV pool** | Another stride-2 conv + `AdaptiveAvgPool2d(8,8)` → flatten → FC → 128-d vector. |
| **Goal fusion** | `Linear(128+2 → 128)` projects the concatenated BEV feature and goal vec. |
| **LSTM-128** | `LSTMCell(128, 128)` maintains temporal state across the episode. GRU is deliberately avoided — PyTorch-MPS GRU has a known performance regression on Apple Silicon. |

The encoder exposes `encode_single()` (single frame, no recurrence) and `forward_seq()` (sequence with LSTM unrolling). Training uses single-step encoding; closed-loop evaluation unrolls the LSTM over the episode.

---

### 2. Vision-Action (VA) Branch

**File:** `bev_vawa/models/va_head.py`

Given the latent `z_t ∈ ℝ¹²⁸`, the VA head scores K=5 **fan anchor waypoints** placed at a fixed horizon (1.5 m) spread across a ±60° arc in front of the robot.

```
z_t  ──►  MLP(128→128→K)  ──►  logits  (B, K)
     ──►  MLP(128→128→K×2) ──►  tanh × 0.3  →  offsets (B, K, 2)  [metres]
```

- **Logits** are trained with cross-entropy against the "best-K" anchor (the one closest to the A\* expert waypoint).
- **Offsets** refine each anchor position by up to ±0.3 m in robot-frame (x, y); trained with Huber loss against the expert waypoint.

Final waypoints: `anchors_k + offset_k`, expressed in the **robot frame** (x-forward, y-left).

---

### 3. World-Action (WA) Branch

**File:** `bev_vawa/models/wa_head.py`

For each candidate waypoint, the WA head rolls out H=3 steps in latent space to predict what would happen if the robot headed toward that candidate.

```
anchor_k (2D robot-frame xy)
   │
   └─► anchor_embed: Linear(2→16→16)  →  e_k  ∈ ℝ¹⁶
              │
              └─► LSTMCell(16, 128) × H steps, initialised from z_t
                              │
                              └─► h_final  (B, K, 128)
                                     ├─► risk_head:     Linear(128→1)   × 3 ensemble members
                                     ├─► progress_head: Linear(128→1)
                                     └─► uncertainty:   Var over sigmoid(risk_ensemble)
```

| Output | Supervision signal | Loss |
|---|---|---|
| `risk_logit` (B, K) | `cand_collision` — whether heading to anchor k hits an obstacle in the rollout window | BCE (+ ensemble BCE) |
| `progress` (B, K) | `cand_progress` — change in distance-to-goal if following anchor k | MSE |
| `uncertainty` (B, K) | No direct supervision — estimated as variance of the 3-member risk ensemble | — |

The LSTMCell (single cell, 128-d hidden) keeps the rollout light: roughly 2×128 extra parameters per ensemble member.

---

### 4. Fusion & Waypoint Selection

**File:** `bev_vawa/models/fusion.py`

```
Q_k = α · softmax(logits)_k   +  β · progress_k
      − γ · sigmoid(risk_logit)_k  −  δ · uncertainty_k
```

Default coefficients (tunable in `configs/default.yaml`):

| Coefficient | Default | Role |
|---|---|---|
| α = 1.0 | policy score — from VA imitation |
| β = 1.5 | forward progress — reward |
| γ = 2.0 | collision risk — penalty (weighted 2× for safety) |
| δ = 0.5 | epistemic uncertainty — soft penalty |

The waypoint with the highest Q is selected and passed to the pure-pursuit controller.

---

### 5. Pure-Pursuit Controller

**File:** `bev_vawa/control/pure_pursuit.py`

Converts the selected waypoint `(wx, wy)` in robot frame to `(v, ω)` commands:

```
heading_err = atan2(wy, wx)
ω  = clip(3.0 × heading_err,  ±max_ang)
v  = max_lin × cos²(heading_err) × min(1, dist / lookahead_min)
```

- Angular velocity is proportional to heading error (P-control).
- Forward speed is tapered by cos² of misalignment, so the robot slows when significantly off-axis — preventing overshoot on tight turns.
- Platform limits: `max_lin = 0.4 m/s`, `max_ang = 1.2 rad/s`.

---

## PIB-Nav Benchmark

**Files:** `bev_vawa/envs/pib_generator.py`, `bev_vawa/envs/occupancy.py`, `bev_vawa/envs/mujoco_env.py`

**PIB-Nav** (Procedural Indoor Benchmark for Navigation) generates random single-room environments on-the-fly in MuJoCo. There are no external scene assets or internet downloads required.

### Room Generation

Each room is sampled from a `RoomSpec`:

| Parameter | Range (default config) |
|---|---|
| Floor dimensions | 5–7 m × 5–7 m |
| Rectangular obstacles | 3–6 obstacles per room |
| Obstacle size | 0.4–1.2 m (half-extent) |
| Wall height | 0.8 m |
| Robot radius | 0.20 m |

The generator builds a MuJoCo XML with:
- Flat floor at z=0
- 4 axis-aligned boundary walls
- N non-overlapping rectangular obstacles (sampled with a 0.3 m clearance to walls)
- A differential-drive base body: two slide joints (X, Y) + one hinge joint (yaw)
- A forward-facing depth camera: 128×128, 90° FOV, 3 m range

MuJoCo's `implicitfast` integrator is used for stability (the default `RK4` integrator diverges with hard contact solvers).

### Occupancy Grid & A\*

`occupancy.py` rasterizes the room XML to a binary grid at 0.05 m/cell (a 7×7 m room → 140×140 grid). Cells occupied by walls or obstacles (including a robot-radius inflation) are marked blocked. An 8-connected A\* with diagonal cost √2 plans a path on this grid. Paths are Chaikin-smoothed (2 iterations) to reduce jagged waypoints.

### NavEnv API

```python
from bev_vawa.envs import NavEnv

env = NavEnv(cfg["env"], seed=0)
obs = env.reset(seed=42)    # new procedural room
# obs keys: depth (128,128), goal_vec (2,), pose (3,)

obs, reward, done, info = env.step(v=0.2, omega=0.5)
# info keys: success, collision, dist_to_goal
```

Episode termination: success when `dist_to_goal < 0.25 m`, or failure on 10 collisions, or 300 steps exceeded.

---

## Data Pipeline

**Files:** `bev_vawa/data/expert.py`, `bev_vawa/data/rollout.py`, `bev_vawa/data/dataset.py`

### Expert Labeling

For each room and start/goal pair:

1. **Plan** an A\* path from start → goal on the occupancy grid.
2. **Smooth** the path with 2 iterations of Chaikin corner cutting.
3. **Resample** the smoothed path to uniform arc-length steps.
4. **Expert waypoint** for the current pose: the path point 1.5 m ahead along the path in robot frame.
5. **Candidate anchors**: K=5 fan points at 1.5 m horizon spanning ±60°.
6. **Best-K index**: the anchor closest (angular distance) to the expert direction.
7. **Candidate labels**: for each anchor, walk its Bresenham raster on the occupancy grid to compute `cand_collision` (hit obstacle in H=3 future steps?) and `cand_progress` (Δ dist-to-goal if taken).

### Rollout

`rollout.py` teleports the robot along the expert path with small Gaussian noise, collecting observations at each step:

| Field | Shape | Description |
|---|---|---|
| `depth` | (128, 128) `float32` | MuJoCo depth render, clipped to 3 m |
| `goal_vec` | (2,) `float32` | Goal minus pose, in robot frame |
| `expert_wp` | (2,) `float32` | Expert next waypoint in robot frame |
| `best_k` | scalar `int64` | Best anchor index |
| `cand_collision` | (K,) `float32` | Binary future collision per anchor |
| `cand_progress` | (K,) `float32` | Normalised Δdist-to-goal per anchor |

Samples are saved as sharded `.npz` files. Full dataset: ~200 rooms × 64 samples ≈ 12,800 samples, ~2 GB on disk.

### Dataset

`NavShardDataset` (subclass of `torch.utils.data.Dataset`) loads shards with a small LRU cache. Each item is a single step. All `.npz` shards are blocked from git by `.gitignore`.

---

## Training

### Two Tracks: Local (MuJoCo) vs Remote (Habitat)

BEV-VAWA-Lite ships with **two training tracks** that share model code, training stages, configs, and the `.npz` shard schema — only the data source differs.

| | **Local (MuJoCo)** | **Remote (Habitat)** |
|---|---|---|
| **Purpose** | Day-to-day development; paper main results | Future research on realistic scenes; cross-domain transfer |
| **Host** | Apple M4 / any machine with PyTorch (MPS/CPU/CUDA) | Linux + CUDA ≥12.1 + EGL (e.g. AWS g5, Lambda Labs, RunPod) |
| **Scenes** | PIB-Nav procedural rooms (built on-the-fly in MuJoCo) | HSSD / HM3D / ProcTHOR-HAB (photorealistic or procedural multi-room) |
| **Install** | `pip install -e .` | `docker build -f docker/habitat.Dockerfile -t bev-vawa-lite:habitat .` |
| **Entry point** | `bash repro.sh --tiny` | `bash docker/train_remote.sh --dataset hssd` |
| **Env adapter** | `bev_vawa/envs/mujoco_env.py::NavEnv` | `bev_vawa/envs/habitat_env.py::HabitatNavEnv` |
| **Data generator** | `scripts/generate_data.py` | `scripts/generate_data_habitat.py` |
| **Evaluator** | `scripts/eval.py` | `scripts/eval_habitat.py` |
| **Configs** | `configs/default.yaml`, `configs/ablations/` | `configs/habitat/{default,hssd,procthor}.yaml` |
| **Time budget** | ~10 min end-to-end | ~3 h on A10 for HSSD |

Both tracks produce checkpoints with the **same architecture**, so you can train on one and evaluate on the other — this is the intended cross-domain generalisation study. See [docker/README.md](docker/README.md) for the remote setup, scene downloads, and cross-domain eval recipes.

The sections below describe the **local track** in detail. Jump to [Remote Track](#remote-track--habitat-scenes-on-a-linux-gpu-box) for the Habitat equivalent.

### Local Track — Stage A/B/C on PIB-Nav (Apple M4)

All training scripts accept `--stage`, `--data`, `--out`, and `--tiny` flags. Configs are loaded from `configs/default.yaml` (or a custom YAML via `--config`). Checkpoints are saved to `runs/<out_dir>/`.

#### Stage A — VA Imitation

**File:** `bev_vawa/train/stage_a_va.py`

Trains the BEV encoder + VA head from scratch to imitate the A\* expert:

```
Loss_A = CE(logits, best_k)  +  0.5 × Huber(sel_waypoint, expert_wp)
```

The cross-entropy term teaches the agent *which* direction to pick; the Huber term refines the spatial accuracy of that waypoint within ±0.3 m. Encoder weights are fully updated.

**Default:** 6 epochs, Adam lr=3e-4, batch=64, gradient clip 1.0.

---

#### Stage B — WA Future Prediction

**File:** `bev_vawa/train/stage_b_wa.py`

Trains the WA head with the encoder **frozen**:

```
Loss_B = BCE(risk_logit, cand_collision)
       + 0.5 × mean(BCE(ensemble_i, cand_collision))   # for all M members
       + MSE(progress, cand_progress)
```

The ensemble members share the same anchor embedding and LSTM rollout weights, but have independent linear output heads. Their variance at test time becomes the uncertainty signal.

**Default:** 6 epochs, same optimiser as Stage A, WA parameters only.

---

#### Stage C — Joint Fine-tuning

**File:** `bev_vawa/train/stage_c_joint.py`

Jointly optimises `Loss_A + Loss_B` with the **last CNN block + LSTM + both heads** unfrozen. The earlier CNN blocks are kept frozen to avoid disrupting low-level feature learning.

**Default:** 3 epochs, lr=1e-4.

---

### Remote Track — Habitat Scenes on a Linux GPU Box

For future research on richer scene datasets (realistic houses, multi-room layouts), a parallel Habitat-sim pipeline is provided. **None of this runs on macOS** — habitat-sim has no EGL headless path on Mac, so this track is Docker-only on a Linux CUDA host.

#### What's new in this track

| Component | File | Purpose |
|---|---|---|
| `HabitatNavEnv` | `bev_vawa/envs/habitat_env.py` | Drop-in replacement for `NavEnv`. Same obs schema (`depth`, `goal_vec`, `pose`), same `(v, ω)` action. Uses habitat-sim's `VelocityControl` + navmesh `try_step()` for collision-aware motion. |
| Habitat rollout | `bev_vawa/data/rollout_habitat.py` | Iterates scene `.glb` files, samples (start, goal) pairs via `pathfinder.find_path`, walks the geodesic path, renders depth, and emits **`.npz` shards with the same schema as the MuJoCo generator**. |
| CLI: data | `scripts/generate_data_habitat.py` | `--scenes 'data/hssd/*.glb' --out data/pib_nav_hssd` |
| CLI: eval | `scripts/eval_habitat.py` | Closed-loop inside Habitat scenes; writes `results/main_table_habitat.csv`. |
| Configs | `configs/habitat/{default,hssd,procthor}.yaml` | Inherit from `configs/default.yaml` and override only what differs (batch size, epochs, scene paths, encoder latent). |
| Container | `docker/habitat.Dockerfile` | CUDA 12.1 + Ubuntu 22.04 + conda `habitat-sim=0.3.2` (headless/EGL) + this package. |
| Remote script | `docker/train_remote.sh` | End-to-end equivalent of `repro.sh` for the GPU box (download scenes → gen data → train A/B/C → eval). |

Because the **shard format is unified**, `scripts/train.py` and all three training stages are **completely unchanged** on the remote track — the model doesn't know whether its training data came from MuJoCo or Habitat.

#### Quickstart (on a Linux + CUDA host)

```bash
# 1. Build the image (~15 min)
docker build -f docker/habitat.Dockerfile -t bev-vawa-lite:habitat .

# 2. Smoke test inside the container
docker run --gpus all --rm -it \
    -v "$PWD":/workspace \
    bev-vawa-lite:habitat \
    bash docker/train_remote.sh --dataset hssd --tiny

# 3. Download an actual scene dataset (inside the container)
python -m habitat_sim.utils.datasets_download \
    --uids hssd-hab --data-path /workspace/data/scene_datasets

# 4. Full pipeline (~3 h on an A10)
bash docker/train_remote.sh --dataset hssd
```

#### Supported scene datasets

| Dataset | Config | Scenes | Character |
|---|---|---|---|
| **HSSD** (Habitat Synthetic Scenes) | `configs/habitat/hssd.yaml` | 200 photorealistic homes | Main Habitat result |
| **ProcTHOR-HAB** | `configs/habitat/procthor.yaml` | 10 000 procedural apartments | Scaling / diversity studies |
| **HM3D** | add new config file | ~1 000 real-world scans | Hardest transfer target |

#### Cross-domain evaluation

The trained checkpoints are **architecture-identical** across tracks, so you can mix and match:

```bash
# Train on PIB-Nav locally, evaluate on HSSD remotely:
python scripts/eval_habitat.py --config configs/habitat/hssd.yaml \
    --scenes 'data/scene_datasets/hssd/val/*.glb' \
    --policy bev_vawa --ckpt runs/default/stage_c.pt \
    --method-name "BEV-VAWA (PIB-Nav → HSSD)"

# Train on HSSD remotely, evaluate on PIB-Nav locally:
python scripts/eval.py --policy bev_vawa \
    --ckpt runs/hssd/stage_c.pt \
    --method-name "BEV-VAWA (HSSD → PIB-Nav)"
```

These transfer numbers go into `results/cross_domain.csv` and are the intended headline result for follow-up work: does a BEV-native architecture generalise from synthetic rooms to realistic houses without retraining?

Full remote-setup guide, troubleshooting, and resource estimates: [docker/README.md](docker/README.md).

---

## Baselines & Ablations

**File:** `bev_vawa/models/baselines.py`, `bev_vawa/train/baseline_trainer.py`

### Baselines (§ Full pipeline)

| Method | Description |
|---|---|
| **A\* Oracle** | GT occupancy map + A\* + pure-pursuit. Upper bound — not a learnable policy. |
| **FPV-BC** | Behavioural cloning from first-person depth: `depth → flatten → MLP → (v, ω)`. No BEV, no candidates. |
| **BEV-BC** | BEV encoder → single MLP head → `(v, ω)`. BEV representation but no candidate reranking. |
| **BEV-VA** | Full VA branch but WA disabled: fusion uses only α·s_k (policy score). |
| **BEV-VAWA (Ours)** | Full model: VA + WA + fusion. |

### Ablations

Configs live in `configs/ablations/` and **inherit** from `configs/default.yaml` to override a single knob:

| Config file | Change | Tests |
|---|---|---|
| `no_wa.yaml` | β=γ=δ=0 (VA-only fusion) | Effect of world model |
| `no_unc.yaml` | δ=0 (remove uncertainty term) | Effect of epistemic uncertainty |
| `h1.yaml` | H=1 rollout horizon | Depth of latent rollout |
| `k1.yaml` | K=1 candidate | Extreme: single waypoint |
| `k3.yaml` | K=3 candidates | Intermediate candidate count |

---

## Evaluation Metrics

**Files:** `bev_vawa/eval/metrics.py`, `bev_vawa/eval/closed_loop.py`

Evaluation runs N=100 held-out episodes (unseen room seeds) and reports:

| Metric | Description |
|---|---|
| **SR** — Success Rate | Fraction of episodes reaching the goal |
| **SPL** — Success weighted by Path Length | SR × (shortest_path / max(agent_path, shortest_path)). Rewards efficient navigation (Anderson et al., 2018). |
| **CollisionRate** | Fraction of episodes with ≥1 collision |
| **PathLenRatio** | For successful episodes: agent path / shortest A\* path. 1.0 = optimal. |
| **LatencyMs** | Mean per-step inference time (model forward pass) in milliseconds |

Results are written to `results/main_table.csv` (main comparison) and `results/ablation_table.csv`.

---

## Install

```bash
# Python 3.10+ recommended.
# torch>=2.4 required to avoid a PyTorch-MPS non-contiguous in-place-op bug
# on older macOS versions.
pip install -e .
```

Core dependencies (pinned in `pyproject.toml`):

| Package | Min version | Reason |
|---|---|---|
| `torch` | ≥ 2.4 | MPS non-contiguous tensor bug fix |
| `mujoco` | ≥ 3.1 | implicitfast integrator + depth sensor |
| `numpy` | any | array operations |
| `pyyaml` | any | config loading |
| `tqdm` | any | progress bars |
| `imageio` | any | depth frame I/O |
| `matplotlib` | any | figures |
| `pytest` | any | stage gate tests |

No CUDA required. The `get_device()` helper in `bev_vawa/utils/device.py` automatically selects MPS → CUDA → CPU in priority order.

---

## Quickstart (tiny, <5 minutes on M4)

```bash
bash repro.sh --tiny
```

This runs the full pipeline in miniature:

1. `pytest tests/` — all 37 stage-gate tests
2. `generate_data.py --tiny` — 5 rooms × 20 samples → `data/pib_nav_tiny/`
3. `train.py --stage a --tiny` — 1 epoch × 3 batches, Stage A
4. `train.py --stage b --tiny` — 1 epoch × 3 batches, Stage B
5. `train.py --stage c --tiny` — 1 epoch × 3 batches, Stage C
6. `eval.py --policy astar --tiny` — 5 episodes, A\* oracle
7. `eval.py --policy bev_vawa --tiny` — 5 episodes, full model
8. `make_figures.py --dummy` — stub figures to `results/`

---

## Full Pipeline

```bash
# Step 1: Generate offline dataset (~200 rooms, ~12k samples, ~2 GB)
python scripts/generate_data.py

# Step 2: Three-stage training
python scripts/train.py --stage a      # imitate A* (VA branch)
python scripts/train.py --stage b      # future prediction (WA branch, encoder frozen)
python scripts/train.py --stage c      # joint fine-tuning

# Step 3: Train baselines
python scripts/train.py --stage fpv_bc
python scripts/train.py --stage bev_bc
python scripts/train.py --stage bev_va

# Step 4: Closed-loop evaluation (100 held-out episodes each)
python scripts/eval.py --policy astar      --method-name "A* Upper Bound"
python scripts/eval.py --policy fpv_bc     --ckpt runs/default/fpv_bc.pt   --method-name FPV-BC
python scripts/eval.py --policy bev_bc     --ckpt runs/default/bev_bc.pt   --method-name BEV-BC
python scripts/eval.py --policy bev_va     --ckpt runs/default/bev_va.pt   --method-name BEV-VA
python scripts/eval.py --policy bev_vawa   --ckpt runs/default/stage_c.pt  --method-name "BEV-VAWA (Ours)"

# Step 5: Figures and paper artefacts
python scripts/make_figures.py
```

Results land in `results/main_table.csv`, `results/ablation_table.csv`, and `results/figures/`.

---

## Configuration System

All hyperparameters live in `configs/default.yaml`. Ablation configs inherit from it and override only the relevant keys:

```yaml
# configs/ablations/no_wa.yaml
inherit: ../default.yaml
fusion:
  alpha: 1.0
  beta: 0.0
  gamma: 0.0
  delta: 0.0   # VA-only: only policy score
```

Key sections in `default.yaml`:

```yaml
env:
  room_size_m:     [5.0, 7.0]   # room dimension range
  n_obstacles:     [3, 6]
  depth_wh:        [128, 128]
  depth_fov_deg:   90
  depth_max_m:     3.0
  goal_tol_m:      0.25
  max_episode_steps: 300

bev:
  grid_size:   64
  latent_dim:  128
  recurrent:   "lstm"           # gru is slower on mps

va:
  n_candidates:       5
  waypoint_horizon_m: 1.5

wa:
  rollout_horizon:    3
  ensemble:           3

fusion:
  alpha: 1.0    # policy score weight
  beta:  1.5    # progress weight
  gamma: 2.0    # risk penalty weight
  delta: 0.5    # uncertainty penalty weight
```

Pass a custom config to any script with `--config path/to/config.yaml`.

---

## Repository Layout

```
bev-vawa-lite/
├── README.md
├── pyproject.toml               # package deps + build metadata
├── repro.sh                     # end-to-end reproducibility script
├── .gitignore                   # blocks data/, runs/, *.pt, *.npz, *.mp4
│
├── configs/
│   ├── default.yaml             # canonical hyperparameters
│   └── ablations/
│       ├── no_wa.yaml           # VA-only fusion (β=γ=δ=0)
│       ├── no_unc.yaml          # remove uncertainty term (δ=0)
│       ├── h1.yaml              # WA rollout horizon H=1
│       ├── k1.yaml              # single candidate K=1
│       └── k3.yaml              # K=3 candidates
│   └── habitat/                 # REMOTE GPU track (see docker/)
│       ├── default.yaml         # habitat base overrides
│       ├── hssd.yaml            # HSSD-200 scene dataset
│       └── procthor.yaml        # ProcTHOR-HAB scene dataset
│
├── bev_vawa/
│   ├── envs/
│   │   ├── pib_generator.py     # procedural MuJoCo XML builder (RoomSpec)
│   │   ├── occupancy.py         # rasterize XML → grid; 8-connected A*
│   │   ├── mujoco_env.py        # NavEnv: reset/step/close; depth + pose
│   │   └── habitat_env.py       # HabitatNavEnv (same API, remote only)
│   │
│   ├── data/
│   │   ├── expert.py            # Chaikin smoothing, candidate anchors, labels
│   │   ├── rollout.py           # MuJoCo: teleport along A*; .npz shards
│   │   ├── rollout_habitat.py   # Habitat: teleport along geodesic; same shard schema
│   │   └── dataset.py           # NavShardDataset (LRU-cached .npz shards)
│   │
│   ├── models/
│   │   ├── bev_encoder.py       # CNN → lift-to-BEV → LSTM → z_t
│   │   ├── va_head.py           # K logits + bounded offsets
│   │   ├── wa_head.py           # LSTMCell rollout → risk / progress / unc
│   │   ├── fusion.py            # fuse_scores() pure function
│   │   ├── full_model.py        # BEVVAWA: orchestrates all modules
│   │   └── baselines.py         # FPV_BC, BEV_BC, BEV_VA
│   │
│   ├── train/
│   │   ├── losses.py            # va_loss(), wa_loss()
│   │   ├── stage_a_va.py        # Stage A trainer
│   │   ├── stage_b_wa.py        # Stage B trainer (frozen encoder)
│   │   ├── stage_c_joint.py     # Stage C joint fine-tuner
│   │   └── baseline_trainer.py  # generic trainer for baselines
│   │
│   ├── control/
│   │   └── pure_pursuit.py      # heading P-control + distance-tapered speed
│   │
│   ├── eval/
│   │   ├── closed_loop.py       # run_episode(), run_eval()
│   │   ├── metrics.py           # spl_score(), summarize()
│   │   └── policies.py          # make_astar_policy(), make_model_policy()
│   │
│   └── utils/
│       ├── seed.py              # set_all(seed) for torch + numpy + random
│       ├── config.py            # load_config() with inherit: support
│       ├── device.py            # get_device() → MPS / CUDA / CPU
│       └── logging.py           # simple run-dir logger
│
├── scripts/
│   ├── generate_data.py          # CLI: build .npz shards (MuJoCo)
│   ├── generate_data_habitat.py  # CLI: build .npz shards (Habitat, remote)
│   ├── train.py                  # CLI: --stage {a,b,c,fpv_bc,bev_bc,bev_va}
│   ├── eval.py                   # CLI: --policy {astar,bev_vawa,...} (MuJoCo)
│   ├── eval_habitat.py           # CLI: same but inside Habitat scenes
│   └── make_figures.py           # CLI: produce all paper figures
│
├── docker/                      # REMOTE GPU track (Habitat)
│   ├── habitat.Dockerfile       # CUDA 12.1 + conda habitat-sim 0.3.2 + project
│   ├── train_remote.sh          # end-to-end pipeline on the GPU box
│   └── README.md                # remote setup, scene download, cross-domain eval
│
├── tests/                       # one pytest file per stage gate
│   ├── test_stage0_env.py
│   ├── test_stage1_env.py
│   ├── test_stage2_data.py
│   ├── test_stage3_va.py
│   ├── test_stage4_wa.py
│   ├── test_stage5_loop.py
│   ├── test_stage6_baselines.py
│   ├── test_stage7_artifacts.py
│   ├── test_stage8_repro.py
│   └── test_stage9_habitat.py   # static checks always run; runtime auto-skips on hosts w/o habitat-sim
│
├── paper/
│   └── paper.md                 # full paper draft (Abstract → Conclusion)
│
└── results/                     # committed CSVs + figures (no checkpoints)
    ├── main_table.csv
    ├── main_table_habitat.csv   # populated by the remote track
    ├── ablation_table.csv
    └── figures/
```

> **What is NOT committed:** `data/` (datasets), `runs/` (checkpoints), `*.pt`, `*.npz`, `*.mp4` — all blocked by `.gitignore`. Only the code, configs, results, and paper draft are in the repo.

---

## Stage Gates

Every stage has a permanent `tests/test_stage*.py` file. Re-running `pytest tests/` at any time checks all prior stages haven't regressed.

| Stage | File | What it checks |
|-------|------|----------------|
| 0 | `test_stage0_env.py` | `torch`, `mujoco` import; MPS conv works; MuJoCo depth render returns finite 64×64 frame |
| 1 | `test_stage1_env.py` | 10 random rooms: A\* finds paths for ≥90% of (start, goal) pairs; 200-step random episode: depth∈[0,3], goal vec consistent with pose |
| 2 | `test_stage2_data.py` | Mini dataset of 100 samples in <30 s; shapes, dtypes, label bounds; Dataset round-trip; collision label balance |
| 3 | `test_stage3_va.py` | Model <3 M params; MPS forward <30 ms; 2-epoch mini-train drives waypoint L2 error below threshold |
| 4 | `test_stage4_wa.py` | 3-epoch mini-train: risk AUROC >0.75, progress Spearman >0.5; fusion returns valid index; full forward <60 ms |
| 5 | `test_stage5_loop.py` | Scripted straight-line episode succeeds; SPL computation verified on hand-crafted toy case; deterministic under fixed seed |
| 6 | `test_stage6_baselines.py` | Each baseline trains 1 epoch without crash; A\* oracle achieves SR≥0.95 on 20 easy rooms |
| 7 | `test_stage7_artifacts.py` | All required figure files exist and non-empty; `main_table.csv` has all method rows; `paper.md` contains all required section headers |
| 8 | `test_stage8_repro.py` | `README.md` contains required sections; `repro.sh` is executable; `.gitignore` blocks heavy outputs; all scripts are syntactically importable; `pyproject.toml` pins `torch>=2.4` |
| 9 | `test_stage9_habitat.py` | Habitat modules import without habitat-sim installed; configs parse; `docker/` files exist and `train_remote.sh` is executable; constructing `HabitatNavEnv` without habitat-sim raises a clear `ImportError`. Runtime tests (actual habitat-sim API) skip on the M4 and run only inside the docker image. |

Run the full suite:

```bash
pytest tests/ -v
# On Apple M4: 43 passed, 1 skipped  (~5 s, no training)
# Inside docker/habitat image: 44 passed  (runtime habitat test active)
```

---

## Engineering Decisions

### Apple Silicon / MPS

- All model code uses `get_device()` — no `.cuda()` calls anywhere.
- `torch>=2.4` is pinned to avoid the PyTorch-MPS non-contiguous in-place tensor bug present in 2.x releases before this version.
- **LSTM over GRU throughout:** PyTorch-MPS has a documented GRU regression. Benchmarks show LSTM is 2–3× faster on M-series chips. The WA rollout uses `LSTMCell` (single-step, no cudnn/mps path dependency).
- Batch size 64 fits comfortably in 16 GB unified memory with the encoder + both heads. `num_workers=0` avoids MPS multiprocessing issues.

### MuJoCo Stability

- Integrator: `implicitfast` (not `RK4`). RK4 diverges with hard contact solvers at the 0.1 s control timestep.
- Contact: `solref="0.005 1"` (soft time constant), `solimp="0.9 0.95 0.001"` — prevents the `Nan/Inf in QACC` warnings that appear with default settings.

### Pose Correctness

MuJoCo slide joints store *displacement from body-home*, not world coordinates. `NavEnv` stores `_start_xy = model.body_pos[body_id, :2]` at construction and adds it in `_pose()` to recover true world coordinates. `teleport()` subtracts it when writing back to `qpos`.

### A\* Robustness

After dynamic motion, the robot may land inside an inflated obstacle cell. `make_astar_policy()` calls `_free_cell()` which BFS-searches for the nearest free cell before planning, ensuring A\* always finds a path.

### Determinism

Every script takes `--seed` and calls `utils.seed.set_all(seed)` which sets `torch.manual_seed`, `np.random.seed`, and Python `random.seed` globally plus seeds the CUDA/MPS RNG if available.

---

## License

Research use only.
