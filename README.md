# BEV-VAWA-Lite

> **Geometry-aware BEV Vision-Action / World-Action navigation for depth-only indoor PointGoal.**
> Trains on a consumer laptop (Apple Silicon / MPS) and scales up to a single-GPU Linux box for Habitat Gibson PointNav v2.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Two Tracks](#two-tracks)
4. [PIB-Nav Benchmark](#pib-nav-benchmark)
5. [Install](#install)
6. [Quickstart](#quickstart-tiny-5-minutes-on-m4)
7. [Full Pipeline (local)](#full-pipeline-local-mujoco)
8. [Full Pipeline (remote Gibson)](#full-pipeline-remote-gibson)
9. [Baselines & Ablations](#baselines--ablations)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Configuration System](#configuration-system)
12. [Repository Layout](#repository-layout)
13. [Stage Gates](#stage-gates)
14. [License](#license)

---

## Overview

**BEV-VAWA-Lite** is a lightweight navigation agent for the indoor PointGoal task: drive a depth-camera robot from a random start pose to a goal position in an unknown room while avoiding obstacles.

The architecture is a single family:

- A **geometry-aware multi-channel BEV encoder** unprojects depth into a local floor plane and produces three channels — occupancy, visible free-space, and a goal-prior sector heatmap — before a small CNN + LSTM distils them into a latent state `z`.
- A **Vision-Action (VA) head** scores `K` candidate waypoints by imitating an A* expert.
- A **latent World-Action (WA) head** rolls each candidate forward for `H` steps and predicts, per candidate: collision risk, progress, ensemble uncertainty, per-step rollout latents `ẑ` (trained against encoder latents via an `L_dyn` loss), and a reachability / dead-end logit.
- A linear **fusion** rule picks the best waypoint:
  ```
  Q_k = α · softmax(s)_k + β · p_k − γ · σ(r_k) − δ · u_k − η · σ(d_k)
  ```
  where `s` = VA logits, `p` = progress, `r` = risk logit, `u` = ensemble uncertainty, `d` = dead-end logit.
- A **pure-pursuit** controller converts the chosen waypoint into `(v, ω)`.

The same model runs on two tracks:

- **Local track** — PIB-Nav, a procedural MuJoCo benchmark. Self-contained, no external assets, trains in 10–20 h total on Apple M4 / MPS.
- **Remote track** — Habitat-sim on Gibson PointNav v2 episodes. Needs a CUDA Linux box; full pipeline finishes in ~3–4 h on a single A10/L4/4090.

Both tracks use the same `BEVVAWA` model, the same Stage A/B/C trainer, and the same closed-loop evaluator.

---

## Architecture

```
depth(H×W)                goal(Δx, Δy)
     │                          │
     ▼                          │
┌─────────────────┐             │
│ GeometryLift    │             │  (pinhole unproject →
│  occ + free +   │             │   scatter into 3 × 64 × 64
│  goal-prior     │             │   metric BEV)
└────────┬────────┘             │
         ▼                      │
   BEV CNN → pool → FC ─concat──┘
         │
         ▼
      LSTMCell  →  z_t  ──────────────────────┐
         │                                     │
         ├───────────────────┐                 │
         ▼                   ▼                 │
    ┌─────────┐        ┌──────────────┐        │
    │  VAHead │        │    WAHead    │        │
    │  s_k,   │        │  r_k,p_k,u_k │        │
    │  waypt  │        │  ẑ_{t+1..H}  │◄── L_dyn vs encode_future(frames)
    │         │        │  d_k         │◄── L_deadend (navmesh label)
    └────┬────┘        └───────┬──────┘        │
         └──────── fuse ───────┘               │
                │                              │
                ▼                              │
           best waypoint k*                    │
                │                              │
                ▼                              │
         pure-pursuit (v, ω) ──► env ─► next depth, next goal ─┘
```

Key files:

| Component | File |
|---|---|
| Differentiable depth → 3-ch BEV lift | `bev_vawa/models/geometry_lift.py` |
| BEV CNN + recurrent encoder | `bev_vawa/models/bev_encoder.py` |
| VA head (candidate scoring + waypoint offsets) | `bev_vawa/models/va_head.py` |
| WA head (risk + progress + ensemble + `ẑ` + dead-end) | `bev_vawa/models/wa_head.py` |
| Fusion rule | `bev_vawa/models/fusion.py` |
| Top-level composition | `bev_vawa/models/full_model.py` |
| Pure-pursuit controller | `bev_vawa/control/pure_pursuit.py` |
| Loss functions (`va_loss`, `wa_loss`) | `bev_vawa/train/losses.py` |

---

## Two Tracks

| Track | Simulator | Dataset | Host | Entry point |
|---|---|---|---|---|
| **Local** | MuJoCo | PIB-Nav (procedural) | Apple M4 / CPU / MPS | `bash repro.sh` |
| **Remote** | Habitat-sim | Gibson PointNav v2 | Linux + CUDA | `bash docker/train_remote.sh` |

Both tracks run the **same `BEVVAWA` model** and the same Stage A/B/C trainer. The remote track's shard format is **schema v2** — each frame additionally stores the next `H` depth frames (`future_depth`) plus a per-candidate reachability label (`cand_deadend`), which the WA head consumes through the `L_dyn` and `L_deadend` losses. Legacy shards without these fields still load; the WA loss gracefully degrades to risk + progress only.

---

## PIB-Nav Benchmark

**PIB-Nav** (Procedural Indoor Benchmark for Navigation) is built entirely in MuJoCo with no external assets:

- Uniform-random rooms `(5–7 m) × (5–7 m)`, walls 0.8 m high.
- 3–6 axis-aligned box obstacles, edge length 0.4–1.2 m.
- Circular agent (radius 0.2 m), differential-drive kinematics, depth camera `128 × 128` at 90° FoV, clipped to 3 m.
- Goal sampled uniformly in a free region ≥ 2 m from the start (straight-line distance) to avoid trivial episodes.
- Success: within 0.25 m of the goal, collisions ≤ 10.

Metrics reported: `SR`, `SPL`, collision rate, path-length ratio, average wall-clock latency per step. See `bev_vawa/eval/metrics.py`.

---

## Install

Requirements:

- Python 3.10+
- PyTorch 2.2+ with MPS (laptop) or CUDA 12.1 (remote).
- `mujoco>=3.1`, `gymnasium`, `pyyaml`, `numpy`, `pytest`, `imageio-ffmpeg` (figure rendering).

For the remote Gibson track you additionally need `habitat-sim==0.3.2` — we ship a Dockerfile that pins everything; see [`docker/README.md`](docker/README.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/        # should end with "72 passed, 2 skipped"
```

---

## Quickstart (tiny, ~5 minutes on M4)

```bash
bash repro.sh --tiny
```

This runs a minimal end-to-end loop: generate 3 small rooms × 6 samples, train Stage A/B/C for 1 epoch each with 3 batches, then evaluate 5 episodes. It exists to verify the environment before committing to the full pipeline.

---

## Full Pipeline (local, MuJoCo)

```bash
bash repro.sh
```

Explicitly:

```bash
# 1. generate dataset (200 rooms × 64 samples ~ 12.8 k frames, ~10 min on M4)
python scripts/generate_data.py --config configs/default.yaml

# 2. three-stage training (~10–20 h total on M4)
python scripts/train.py --config configs/default.yaml --stage a \
       --data data/pib_nav --out runs/default
python scripts/train.py --config configs/default.yaml --stage b \
       --data data/pib_nav --out runs/default
python scripts/train.py --config configs/default.yaml --stage c \
       --data data/pib_nav --out runs/default

# 3. closed-loop evaluation (100 held-out episodes)
python scripts/eval.py --config configs/default.yaml \
       --policy bev_vawa --ckpt runs/default/stage_c.pt \
       --method-name "BEV-VAWA (PIB-Nav)"
```

Results are appended to `results/main_table.csv`.

**Stage A — VA imitation.** Cross-entropy on candidate logits + Huber on the selected waypoint offset.
**Stage B — WA supervision.** Encoder frozen; WA head learns risk/progress/ensemble plus `L_dyn` (MSE between `ẑ_{t+τ}[best_k]` and a no-grad encoder pass over `future_depth`) and `L_deadend` (BCE against the reachability label). When shards carry no future frames (e.g. MuJoCo rollouts), the latter two terms degrade to zero automatically.
**Stage C — Joint fine-tune.** Unfreeze `bev_pool`, `fc_pool`, and `input_proj`; `λ_dyn` and `λ_deadend` drop by 0.4× to keep the dynamics loss from destabilising the encoder.

---

## Full Pipeline (remote, Gibson)

Prerequisites: a Linux host with CUDA 12.1, the NVIDIA Container Toolkit, and a Gibson / PointNav-v2 EULA on file at aihabitat.org. See [`docker/README.md`](docker/README.md) for the full setup.

```bash
# one-time: build the image
docker build -f docker/habitat.Dockerfile -t bev-vawa-lite:habitat .

# one-time: download Gibson scenes + PointNav v2 episodes (inside container)
python -m habitat_sim.utils.datasets_download --uids gibson            --data-path data/scene_datasets
python -m habitat_sim.utils.datasets_download --uids pointnav_gibson_v2 --data-path data

# smoke test (<5 min on a single A10)
bash docker/train_remote.sh --tiny

# full pipeline (~3–4 h on a single A10/L4/4090)
bash docker/train_remote.sh
```

`docker/train_remote.sh` runs, in order:

1. `torch.cuda.is_available()` + `habitat_sim` import sanity check.
2. Pytest smoke (static checks only).
3. Schema-v2 shard generation for `train` and `val` splits (`scripts/generate_data_habitat.py`).
4. Stage A → Stage B → Stage C (`scripts/train.py`, same trainer as the local track).
5. Closed-loop evaluation on the val split (`scripts/eval_habitat.py`).

A row is appended to `results/main_table_habitat.csv` with method `"BEV-VAWA (Gibson)"`.

---

## Baselines & Ablations

All baselines live in `bev_vawa/models/baselines.py` and reuse the same `BEVEncoder`:

| Model | What it ablates |
|---|---|
| `FPV_BC` | A ResNet-style first-person-view-to-waypoint regressor — removes the BEV lift entirely. |
| `BEV_BC` | BEV encoder + direct waypoint regression (no K-candidate head). |
| `BEV_VA` | BEV encoder + VA head only (no WA branch). |
| `BEVVAWA` | Full model (main method). |

Ablation configs under `configs/ablations/`:

| Config | Toggle |
|---|---|
| `no_wa.yaml` | `fusion.{β,γ,δ,η}=0` — VA-only fusion at inference. |
| `no_unc.yaml` | `fusion.δ=0` — drop the ensemble-uncertainty term. |
| `h1.yaml` | `wa.rollout_horizon=1` — WA collapses to single-step prediction. |
| `k1.yaml` / `k3.yaml` | `va.n_candidates` = 1 / 3. |

Gibson-only ablation:

| Config | Toggle |
|---|---|
| `configs/habitat/gibson_occonly.yaml` | `bev.channels_enabled=[1,0,0]` — multi-channel BEV collapses to occupancy-only. |

Each `.yaml` is a thin override on top of `configs/default.yaml` via the `inherit:` chain.

---

## Evaluation Metrics

Computed by `bev_vawa/eval/metrics.py::summarize`:

- **SR** — success rate over `n_episodes`.
- **SPL** — success weighted by (shortest / max(shortest, agent-path)).
- **CollisionRate** — mean collisions per episode.
- **PathLenRatio** — mean `agent / shortest` on successes.
- **LatencyMs** — mean wall-clock time per policy call.

Evaluators: `scripts/eval.py` (MuJoCo) and `scripts/eval_habitat.py` (Habitat). Both write a row per method to a single CSV so side-by-side tables are a simple pandas read.

---

## Configuration System

YAML files under `configs/` use an `inherit:` chain with deep-merge:

```
configs/default.yaml                        # root (PIB-Nav / MuJoCo defaults)
├── configs/ablations/*.yaml                # inherits ../default.yaml
└── configs/habitat/default.yaml            # inherits ../default.yaml
    ├── configs/habitat/gibson.yaml         # inherits ./default.yaml
    └── configs/habitat/gibson_occonly.yaml # inherits ./gibson.yaml
```

`bev_vawa/utils/config.py::load_config` resolves the chain and returns a plain `dict`. Any scalar / nested key in a child overrides the parent; lists are replaced wholesale.

Every model constructor (e.g. `BEVVAWA(cfg)`) takes that dict and reads only the sections it needs (`env`, `bev`, `va`, `wa`) — so a single YAML drives model shape, env parameters, training hyperparameters, and dataset paths.

---

## Repository Layout

```
bev_vawa/
├── control/              pure_pursuit.py
├── data/                 expert labels, MuJoCo & Habitat rollouts,
│                         Gibson episode loader, NavShardDataset (schema-aware)
├── envs/                 MuJoCoNavEnv, HabitatNavEnv, occupancy utils
├── eval/                 policies.py (make_model_policy), metrics.py
├── models/               geometry_lift, bev_encoder, va_head,
│                         wa_head, fusion, full_model, baselines
├── train/                stage_{a,b,c}, baseline_trainer,
│                         losses (va_loss, wa_loss), _common (build_model,
│                         wa_loss_for_stage)
└── utils/                config loader, device, seeds

configs/
├── default.yaml          root config (PIB-Nav / MuJoCo)
├── ablations/            no_wa, no_unc, h1, k1, k3
└── habitat/              default, gibson, gibson_occonly

docker/
├── habitat.Dockerfile    CUDA 12.1 + conda habitat-sim 0.3.2 + bev_vawa
├── train_remote.sh       end-to-end Gibson pipeline
└── README.md             remote-track setup + troubleshooting

scripts/
├── generate_data.py              MuJoCo data generator
├── generate_data_habitat.py      Gibson v2 data generator (schema v2)
├── train.py                      Stage A / B / C (stage via --stage)
├── eval.py                       MuJoCo closed-loop eval
├── eval_habitat.py               Gibson closed-loop eval
└── make_figures.py               diagnostic plots

tests/                            72 passing on macOS (2 habitat runtime skips)
repro.sh                          local pipeline
```

---

## Stage Gates

Each `tests/test_stage*.py` file is a gate: a specific invariant must hold before the next stage can land. On a laptop without habitat-sim, stage 9 runtime tests skip; all other stages run in ~6 s on M4.

| Gate | What it locks in |
|---|---|
| stage 0 | MuJoCo env constructs and steps. |
| stage 1 | Depth/goal observations well-formed. |
| stage 2 | Expert anchors + shard writer produce a loadable dataset. |
| stage 3 | `BEVVAWA` forward shape + size + Stage-A smoke. |
| stage 4 | WA head forward + fusion sanity + Stage-B/C smoke. |
| stage 5 | Closed-loop policy runs end-to-end on MuJoCo. |
| stage 6 | Baselines + ablation configs load + one forward pass each. |
| stage 7 | `scripts/make_figures.py` emits all expected artefacts. |
| stage 8 | `repro.sh --tiny` finishes green. |
| stage 9 | Habitat scaffolding compiles on every host; runtime gated on `habitat_sim`. |
| stage 10 | `GeometryLift` occupancy / free-space / goal-prior semantics; `BEVEncoder` shapes + gradient flow. |
| stage 11 | `WAHead` per-step `ẑ` + dead-end logit; `wa_loss` terms; end-to-end backward through `BEVVAWA`. |
| stage 12 | Gibson episode loader + schema-v2 shard detection + pipeline wiring static checks. |

Run the full gate suite:

```bash
pytest tests/ -v
```

---

## License

MIT. See `LICENSE`.
