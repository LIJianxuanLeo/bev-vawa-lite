# Remote Habitat Training — Gibson PointNav v2

This directory contains everything needed to run the **remote track** of
BEV-VAWA-Lite on a Linux GPU host using Habitat-sim on the Gibson
PointNav v2 episode pack. The local MuJoCo workflow at the repo root
(`repro.sh`, `configs/default.yaml`) is untouched — the two tracks share
the same `BEVVAWA` model and Stage A/B/C trainer, they only differ in
which simulator drives the data-gen and closed-loop eval.

| Track | Simulator | Dataset | Host | Entry point |
|---|---|---|---|---|
| **Local** | MuJoCo | PIB-Nav (procedural) | Apple M4 / CPU / MPS | `bash repro.sh` |
| **Remote** | Habitat-sim | Gibson PointNav v2 | Linux + CUDA | `bash docker/train_remote.sh` |

Gibson shards are written in **schema v2** (adds `future_depth`,
`future_goal`, `cand_deadend`, `schema_version: 2`), which feeds the WA
head's `L_dyn` (latent-dynamics alignment) and `L_deadend` (reachability)
losses. When shards lack those keys — e.g. local MuJoCo rollouts — the
two losses degrade to zero automatically and the WA head falls back to
the classical risk + progress supervision.

---

## 1. Hardware & OS requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA CUDA 12.1, ≥ 12 GB VRAM | A10 / A100 / L4 / RTX 4090 |
| OS | Ubuntu 22.04 (via Docker) | same |
| Disk | 60 GB (Gibson scenes + v2 shards + checkpoints) | 120 GB |
| Docker | Docker 24+ with the NVIDIA Container Toolkit | |

macOS is **not** supported here — habitat-sim has no EGL headless path on
macOS. Use the MuJoCo track at the repo root instead.

---

## 2. Build the image

From the repo root:

```bash
docker build -f docker/habitat.Dockerfile -t bev-vawa-lite:habitat .
```

Build takes ~15 min (conda-installing habitat-sim is the slow part). The
resulting image is ~8 GB. The final `RUN` step does an import check, so
a successful build implies `torch.cuda.is_available() == True` and
`habitat_sim` imports cleanly inside the container.

---

## 3. Download scene assets

Gibson scenes and PointNav v2 episodes are gated on Stanford's aihabitat
mirror and require a **one-time EULA** acceptance at
<https://aihabitat.org/datasets/gibson/>. After that, inside the
running container:

```bash
docker run --gpus all --rm -it \
    -v "$PWD":/workspace \
    bev-vawa-lite:habitat bash

# Gibson scene meshes (.glb)
python -m habitat_sim.utils.datasets_download \
    --uids gibson \
    --data-path /workspace/data/scene_datasets

# PointNav v2 episodes (.json.gz)
python -m habitat_sim.utils.datasets_download \
    --uids pointnav_gibson_v2 \
    --data-path /workspace/data

# Result:
#   /workspace/data/scene_datasets/gibson/*.glb                         (scenes)
#   /workspace/data/datasets/pointnav/gibson/v2/{train,val}/*.json.gz   (episodes)
```

The loader at `bev_vawa/data/gibson_episodes.py::iter_episodes`
understands both the flat `{split}/{split}.json.gz` layout and the
per-scene `{split}/content/*.json.gz` layout that the v2 val split uses.

---

## 4. Run the full pipeline

```bash
docker run --gpus all --rm -it \
    -v "$PWD":/workspace \
    -v "$PWD/data":/workspace/data \
    -v "$PWD/runs":/workspace/runs \
    -v "$PWD/results":/workspace/results \
    bev-vawa-lite:habitat \
    bash docker/train_remote.sh
```

`train_remote.sh` runs, in order, inside the container:

1. Sanity: `torch.cuda` + `habitat_sim` import check.
2. Pytest static gates.
3. Schema-v2 shard generation for `train` + `val` splits via
   `scripts/generate_data_habitat.py`.
4. Stage A → Stage B → Stage C via the unified `scripts/train.py`.
5. Closed-loop eval on the val split via `scripts/eval_habitat.py`,
   appending a row to `results/main_table_habitat.csv`.

**Smoke test first.** Always run `--tiny` once to validate the environment
before kicking off a full training run:

```bash
bash docker/train_remote.sh --tiny
```

Tiny mode uses 2 scenes × 2 episodes × 2 samples, 1 training epoch with a
handful of batches, and 5 eval episodes — finishes in under 5 minutes on
an A10.

---

## 5. Expected resource usage

Rough measurements on a single A10 (24 GB VRAM), ~60 Gibson train scenes:

| Phase | Time | Notes |
|---|---|---|
| Data generation | ~1 h | Includes H=3 future-depth capture (teleport-forward, render, teleport-back) for `L_dyn`. |
| Stage A (20 epochs) | ~40 min | Encoder + VA head only. |
| Stage B (20 epochs) | ~50 min | Encoder frozen; WA head with `L_dyn` + `L_deadend` supervision. |
| Stage C (8 epochs) | ~25 min | Unfreeze `bev_pool` / `fc_pool` / `input_proj`; `λ_dyn` / `λ_deadend` × 0.4. |
| Evaluation (200 eps) | ~30 min | Closed loop in Habitat. |
| **Total** | **~3.5 h** | |

VRAM peaks at ~8 GB (batch 128, depth 128 × 128, latent 128, plus the
no-grad encoder pass over future frames).

Data-gen dominates because each sampled step also renders an `H`-step
future-depth stack. Knobs in `configs/habitat/gibson.yaml`:

- `gibson.samples_per_episode` — how many samples to take per expert path.
- `gibson.rollout_horizon` — `H` for future-depth capture (must match
  `wa.rollout_horizon`; default 3).
- `gibson.max_episodes_per_scene` / `gibson.scene_limit` — caps for quick
  debugging or paper-scale runs.

---

## 6. Cross-domain evaluation

Because the local and remote tracks share the same `BEVVAWA` model,
checkpoints transfer between them:

```bash
# Gibson-trained model, evaluated on PIB-Nav (MuJoCo)
python scripts/eval.py --config configs/default.yaml \
    --policy bev_vawa \
    --ckpt   runs/gibson/stage_c.pt \
    --method-name "BEV-VAWA (Gibson → PIB-Nav)"

# PIB-Nav-trained model, evaluated on Gibson (Habitat)
python scripts/eval_habitat.py --config configs/habitat/gibson.yaml \
    --scenes 'data/scene_datasets/gibson/*.glb' \
    --policy bev_vawa \
    --ckpt   runs/default/stage_c.pt \
    --method-name "BEV-VAWA (PIB-Nav → Gibson)"
```

Both writers append to `results/main_table_habitat.csv` (Habitat side)
and `results/main_table.csv` (MuJoCo side), so cross-domain tables are a
simple pandas read.

---

## 7. File map

```
docker/
├── habitat.Dockerfile   CUDA 12.1 + conda habitat-sim 0.3.2 + bev_vawa
├── train_remote.sh      end-to-end Gibson pipeline (data → train → eval)
└── README.md            this file

bev_vawa/
├── envs/habitat_env.py           HabitatNavEnv (drop-in for the MuJoCo env)
├── data/rollout_habitat.py       Habitat shard generator (schema v2)
├── data/gibson_episodes.py       PointNav v2 episode loader (.json.gz)
└── models/                       shared geometry-aware BEV + WA head

scripts/
├── generate_data_habitat.py      CLI wrapper for rollout_habitat
└── eval_habitat.py               closed-loop eval inside Habitat scenes

configs/habitat/
├── default.yaml                  base habitat overrides
├── gibson.yaml                   Gibson PointNav v2 config
└── gibson_occonly.yaml           ablation: BEV occupancy channel only

tests/
├── test_stage9_habitat.py        habitat scaffolding compiles (auto-skip without sim)
├── test_stage10_bev_encoder.py   geometry-lift + encoder shapes / gradients
├── test_stage11_wa_head.py       WA head + L_dyn + L_deadend losses
└── test_stage12_gibson.py        Gibson episode loader + schema-v2 wiring
```

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: habitat-sim is not available` | You ran habitat code outside the docker image. Either rebuild/relaunch the container, or restrict yourself to the MuJoCo track. |
| `no navmesh loaded for scene ...` | The scene `.glb` ships without a `.navmesh`. Gibson scenes include one; if you regenerate the mesh yourself, recompute via `sim.recompute_navmesh(habitat_sim.NavMeshSettings())`. |
| `libEGL.so.1: cannot open shared object` | Docker wasn't launched with `--gpus all`, or the NVIDIA Container Toolkit isn't installed on the host. |
| Depth frames are all zeros | Common symptom of a disabled depth sensor on some drivers; check `sim_cfg.gpu_device_id` matches a real CUDA index. |
| `CUDA out of memory` during training | Drop `train.batch_size` (default 128 → 64) in `configs/habitat/gibson.yaml`. |
| Data-gen is too slow | Drop `gibson.rollout_horizon` 3 → 1 (disables per-step future-depth capture, but `L_dyn` loses its supervision signal — expect WA to regress to risk + progress only). |
