# Remote Habitat Training

This directory contains everything needed to run the **Habitat-sim track** of
BEV-VAWA-Lite on a Linux GPU host. The local Apple M4 workflow in the repo
root (`repro.sh`, `configs/default.yaml`) is untouched — the remote track is
purely additive for follow-up research on realistic 3D scene datasets.

| Track | Where it runs | What it trains on | Entry point |
|---|---|---|---|
| **Local (MuJoCo)** | Apple M4 / CPU / MPS | PIB-Nav procedural rooms | `bash repro.sh --tiny` |
| **Remote (Habitat)** | Linux + CUDA + EGL | HSSD / HM3D / ProcTHOR-HAB | `bash docker/train_remote.sh --dataset hssd` |

Both tracks share **the same model code, training stages, config grammar,
and shard schema** — only the simulator and scene dataset differ.

---

## 1. Hardware & OS requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA CUDA 12.1, ≥12 GB VRAM | A10 / A100 / RTX 4090 |
| OS | Ubuntu 22.04 (via Docker) | same |
| Disk | 80 GB (scenes + shards + checkpoints) | 250 GB for HSSD full |
| Docker | Docker 24+ with the NVIDIA Container Toolkit | |

macOS is **not** supported for this track — habitat-sim has no EGL headless
path on macOS. If you only have a Mac, run the local MuJoCo track instead.

---

## 2. Build the image

From the repo root:

```bash
docker build -f docker/habitat.Dockerfile -t bev-vawa-lite:habitat .
```

The build step takes ~15 min (conda installing habitat-sim is the slow part).
The resulting image is ~8 GB.

**Verification** — the Dockerfile's final step runs an import check; if the
build succeeds you already know that `torch.cuda.is_available() == True` and
`habitat_sim` imports cleanly inside the container.

---

## 3. Download scene datasets

Scene assets are **not** shipped with the repo. Inside the running container:

```bash
# Inside container (after `docker run --gpus all -it bev-vawa-lite:habitat`):

# HSSD (Habitat Synthetic Scenes Dataset, 200 scenes, ~20 GB)
python -m habitat_sim.utils.datasets_download \
    --uids hssd-hab \
    --data-path /workspace/data/scene_datasets

# OR ProcTHOR-HAB (10k procedural apartments, smaller per-scene but more scenes)
python -m habitat_sim.utils.datasets_download \
    --uids procthor-hab-10k \
    --data-path /workspace/data/scene_datasets

# OR HM3D (photorealistic, requires Matterport licence acceptance first)
python -m habitat_sim.utils.datasets_download \
    --uids hm3d_train_v0.2 hm3d_val_v0.2 \
    --data-path /workspace/data/scene_datasets
```

HSSD and HM3D are gated on HuggingFace / Matterport and require a token;
follow the habitat-lab docs for credentials.

---

## 4. Run the full pipeline

```bash
docker run --gpus all --rm -it \
    -v "$PWD":/workspace \
    -v "$PWD/data":/workspace/data \
    -v "$PWD/runs":/workspace/runs \
    -v "$PWD/results":/workspace/results \
    bev-vawa-lite:habitat \
    bash docker/train_remote.sh --dataset hssd
```

This runs inside the container:

1. Dependency sanity check (`torch.cuda`, `habitat_sim.__version__`).
2. Regenerate `.npz` shards from HSSD scenes with `generate_data_habitat.py`.
3. Stage A → B → C training with `train.py` (unchanged, reads shards).
4. Closed-loop evaluation on the val split with `eval_habitat.py`.
5. CSV row appended to `results/main_table_habitat.csv`.

**Smoke test first:** always run `--tiny` once to validate the environment
before kicking off a full training run:

```bash
bash docker/train_remote.sh --dataset hssd --tiny
```

Tiny mode uses 1 scene × 2 pairs × 4 samples, 1 training epoch with 3 batches,
and 5 eval episodes — finishes in under 2 minutes on an A10.

---

## 5. Expected resource usage

Rough measurements on a single A10 (24 GB VRAM):

| Phase | HSSD (full) | ProcTHOR-HAB (10k) |
|---|---|---|
| Data generation | ~40 min (200 scenes × 24 pairs × 12 steps) | ~2 h (10k scenes × 4 pairs × 4 steps) |
| Stage A (30 epochs) | ~50 min | ~4 h |
| Stage B (30 epochs) | ~50 min | ~4 h |
| Stage C (10 epochs) | ~20 min | ~1.5 h |
| Evaluation (200 eps) | ~30 min | ~30 min |
| **Total** | **~3 h** | **~12 h** |

The dataset size is the main driver. VRAM usage peaks at ~10 GB during
training (batch 128, depth 128×128, encoder latent 192).

---

## 6. Cross-domain evaluation

The trained checkpoints are architecture-identical to the MuJoCo-trained
ones, so you can cross-evaluate:

```bash
# HSSD-trained model, evaluated on PIB-Nav (laptop, MuJoCo)
python scripts/eval.py --config configs/default.yaml \
    --policy bev_vawa \
    --ckpt   runs/hssd/stage_c.pt \
    --method-name "BEV-VAWA (HSSD → PIB-Nav)"

# PIB-Nav-trained model, evaluated on HSSD (remote, Habitat)
python scripts/eval_habitat.py --config configs/habitat/hssd.yaml \
    --scenes data/scene_datasets/hssd/val/*.glb \
    --policy bev_vawa \
    --ckpt   runs/default/stage_c.pt \
    --method-name "BEV-VAWA (PIB-Nav → HSSD)"
```

The transfer numbers go into `results/cross_domain.csv` and are the intended
headline result for the follow-up paper — they test whether a BEV-native
architecture generalises from synthetic rooms to photorealistic scenes
without retraining.

---

## 7. File map

```
docker/
├── habitat.Dockerfile   # CUDA 12.1 + conda habitat-sim 0.3.2 + bev_vawa
├── train_remote.sh      # end-to-end pipeline (data → train → eval)
└── README.md            # this file

bev_vawa/envs/habitat_env.py        # HabitatNavEnv (drop-in for NavEnv)
bev_vawa/data/rollout_habitat.py    # habitat shard generator (same schema)
scripts/generate_data_habitat.py    # CLI for the above
scripts/eval_habitat.py             # closed-loop eval inside habitat scenes
configs/habitat/
├── default.yaml         # base habitat overrides
├── hssd.yaml            # HSSD-specific
└── procthor.yaml        # ProcTHOR-HAB-specific
tests/test_stage9_habitat.py   # stage gate (auto-skips on hosts w/o habitat-sim)
```

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: habitat-sim is not available` | You ran habitat code outside the docker image. Either rebuild/relaunch the container, or restrict yourself to the MuJoCo track on this host. |
| `no navmesh loaded for scene ...` | The scene .glb has no corresponding `.navmesh`. Recompute with `sim.recompute_navmesh(habitat_sim.NavMeshSettings())`, or use HSSD/HM3D scenes that ship with navmeshes. |
| `libEGL.so.1: cannot open shared object` | Docker wasn't launched with `--gpus all`, or the NVIDIA Container Toolkit isn't installed on the host. |
| Depth frames are all zeros | Common symptom of a disabled depth sensor on some drivers; check `sim_cfg.gpu_device_id` matches a real CUDA index. |
| `CUDA out of memory` during training | Drop `train.batch_size` (default 128 → 64) in `configs/habitat/<dataset>.yaml`. |
