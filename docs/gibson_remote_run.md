# Gibson PointNav v2 — Remote Training Runbook

This document captures the concrete steps, pitfalls, and numbers from the
first end-to-end Gibson Habitat run of BEV-VAWA-Lite on a 1×RTX 4090 cloud
node (2026-04-22). It is meant to let the next operator reach a trained
checkpoint in **under 3 hours of billable GPU time, without repeating any
of the debugging loops that cost us ~1.5 h the first time around**.

The numbers below are for the Habitat Gibson 4+ subset (~86 scenes, not
the 10 GB full `gibson_habitat_trainval`). If you need full Gibson, see
§5 "Full trainval pack" at the end.

---

## 1. What you sign, what you download

### 1.1 EULA (one-time, per-person)

1. Go to <https://aihabitat.org/datasets/gibson/>.
2. Fill out the Stanford Gibson Database form. Approval email arrives in
   5 min–1 h with a set of plain `https://dl.fbaipublicfiles.com/...` or
   `https://storage.googleapis.com/gibson_scenes/...` URLs — **no
   username/password**, the URLs themselves are the access token.
3. Save the email; the URLs do not rotate.

### 1.2 Files we actually need (minimal)

| File | Size | Purpose |
|---|---|---|
| [`gibson_habitat.zip`](https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip) | **1.40 GiB** | Habitat 4+ subset: 86 `.glb` meshes + `.navmesh` + `.scn` |
| [`pointnav_gibson_v2.zip`](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip) | **274 MiB** | PointNav v2 episode pack (train / val / val\_mini) |

The full `gibson_habitat_trainval.zip` (~10 GiB) is NOT required unless you
specifically need the "2+" lower-quality scenes. The 4+ subset covers all
PointNav v2 episodes in both train and val splits.

---

## 2. Known-good download procedure (**do not use aria2c**)

### 2.1 What fails

- **`aria2c -x16`** from `dl.fbaipublicfiles.com`: cold-CDN edge nodes
  rate-limit parallel segments to ~100 KiB/s total, producing a 10 GiB
  ETA of 26 hours and a ~1-in-2 chance of leaving a corrupt zip with
  missing chunks (it reports "ETA complete" but `unzip -t` reports
  `End-of-central-directory signature not found`).
- `curl --range` speed benchmarks **do not** predict sustained throughput
  for the full zip; edge caching behavior is byte-range-specific.

### 2.2 What works

Single-threaded `wget` with content-length verification:

```bash
cd /root/data/downloads
wget --progress=dot:giga --tries=3 --timeout=60 \
    https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip

# Verify against remote content-length, BEFORE unzipping
ACTUAL=$(stat -c%s gibson_habitat.zip)
EXPECTED=$(curl -sI https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip \
             | awk '/content-length/ {print $2}' | tr -d '\r')
[ "$ACTUAL" = "$EXPECTED" ] || { echo "FAIL: $ACTUAL != $EXPECTED"; exit 1; }

# Integrity check before extraction
unzip -t gibson_habitat.zip > /dev/null && echo OK
```

Expected throughput from mainland China cloud GPU nodes to this CDN:
**1–6 MiB/s sustained**, i.e. 4–20 min for the 1.4 GiB scene pack. The
episode pack finishes in ~30 s.

### 2.3 Unzip target layout

```
/root/data/
├── scene_datasets/gibson/          # from gibson_habitat.zip
│   ├── Adrian.glb, Adrian.navmesh, Adrian.scn
│   ├── ...
│   └── (86 scenes total)
└── datasets/pointnav/gibson/v2/    # from pointnav_gibson_v2.zip
    ├── train/train.json.gz
    ├── train/content/{Adrian,...}.json.gz   # 72 files
    ├── val/val.json.gz
    ├── val/content/{Cantwell,...}.json.gz   # 14 files
    └── val_mini/...                          # 3 files
```

The 72 train scenes and 14 val scenes in the episode pack are all present
as `.glb` in the scene pack (100% coverage, pre-checked with
`comm -12 ep_scenes.txt glb_scenes.txt`).

---

## 3. Environment setup — **habitat-sim only ships py3.9 wheels**

### 3.1 The py3.9 constraint

As of 0.3.3, the `aihabitat` conda channel publishes habitat-sim
`headless_bullet_linux` wheels **only for Python 3.9**. The ebcloud /
Alibaba / most cloud PyTorch base images ship Python 3.10+, so you MUST
create a dedicated conda env.

```bash
conda create -y -n habitat python=3.9
conda install -y -n habitat -c conda-forge -c aihabitat \
    habitat-sim=0.3.3 withbullet=1 headless=1

HP=/root/miniconda3/envs/habitat/bin

# Torch 2.7 + CUDA 12.6 matches RTX 4090 driver 580.x
$HP/pip install --no-cache-dir torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Project deps. pillow MUST be pinned to habitat-sim's exact version
# (it complains on mismatch and silently falls back to a software JPEG
# decoder that halves the data-gen speed).
$HP/pip install --no-cache-dir \
    "pillow==10.4.0" numpy pyyaml tqdm imageio imageio-ffmpeg \
    matplotlib tyro "mujoco==3.1.6"  # mujoco only for import compatibility
```

Note the `mujoco==3.1.6` pin: this is the last version with a py3.9 wheel
on PyPI. Newer mujoco builds require `MUJOCO_PATH` + source compile, which
fails in containerized setups. We only use mujoco to satisfy the import
chain in `bev_vawa/envs/__init__.py`; none of its functionality is used
on the Habitat track.

### 3.2 Project install

```bash
# pyproject.toml requires python >=3.10; override for the habitat env
sed -i 's/requires-python = ">=3.10"/requires-python = ">=3.9"/' pyproject.toml
$HP/pip install --no-cache-dir -e . --no-deps
```

(This edit is in-Pod only and not committed; the local MuJoCo/PIB-Nav
track is still Python 3.10+.)

---

## 4. Performance-critical trainer knobs

These three patches together delivered a **~80× epoch-time speedup**
(5.5 min → 3.7 sec) on the Gibson train shards:

| File | Change | Why |
|---|---|---|
| `bev_vawa/data/dataset.py` | `_cache` LRU size 8 → 128 | Covers all 72 shards per worker; avoids re-parsing `.npz` on every `__getitem__` cache miss. |
| `bev_vawa/train/stage_{a,b,c}.py` | `persistent_workers=True, pin_memory=True, prefetch_factor=4` | Prevents workers being respawned each epoch (which would nuke the shard cache). |
| `configs/habitat/gibson.yaml` | `train.num_workers: 4 → 8` | Cloud 4090 nodes usually have 8–16 cores; the 4-worker default bottlenecks Python-side shard parsing. |

The symptom without these patches is **GPU utilisation stays at 0–19%**
while **every DataLoader worker pins a CPU core at 99%**. `nvidia-smi`
and `ps aux --sort=-%cpu` make this obvious in ~10 seconds of inspection.

Additionally, staging the shards on `/dev/shm` (tmpfs) gives **no
measurable speedup** on top of the cache patch, because the bottleneck
is Python-side `np.load` parsing, not raw disk read bandwidth. Save the
RAM for other uses.

---

## 5. Habitat-sim 0.3.2 → 0.3.3 API break

`AgentState.rotation` in 0.3.3 returns a `numpy-quaternion` object; the
`.rigid_state()` method on `AgentState` is gone. The old code path

```python
state = self._agent.state
new_rigid = self._vel_ctrl.integrate_transform(
    self.control_dt, state.rigid_state()
)
```

is now a hard `AttributeError`. The drop-in replacement (see
`bev_vawa/envs/habitat_env.py::HabitatNavEnv.step`):

```python
import magnum as _mn
import quaternion as _qt

_rs = habitat_sim.RigidState()
_p = state.position
_rs.translation = _mn.Vector3(float(_p[0]), float(_p[1]), float(_p[2]))
_q = state.rotation
_rs.rotation = _mn.Quaternion(
    _mn.Vector3(float(_q.x), float(_q.y), float(_q.z)), float(_q.w)
)
new_rigid = self._vel_ctrl.integrate_transform(self.control_dt, _rs)
# ... after try_step / collision snap ...
_r = new_rigid.rotation
new_state.rotation = _qt.quaternion(
    float(_r.scalar), float(_r.vector.x),
    float(_r.vector.y), float(_r.vector.z),
)
```

Two subtle points:

1. **Explicit `magnum.Vector3(float(...), ...)`** — assigning a numpy
   array to `RigidState.translation` appears to succeed but silently
   stores a zero vector, producing an "agent teleports to origin each
   step" failure mode that presents as 100% collision rate.
2. **Explicit numpy-quaternion round-trip on write-back** — assigning
   the raw `magnum.Quaternion` to `AgentState.rotation` raises
   `TypeError: incompatible function arguments` at runtime.

---

## 6. End-to-end command sequence (copy-paste)

Assumes you already have the two zips downloaded to `/root/data/downloads/`
and the habitat conda env from §3.

```bash
# Unpack
cd /root/data/scene_datasets && unzip -oq /root/data/downloads/gibson_habitat.zip
cd /root/data/datasets/pointnav/gibson/v2&& \
    unzip -oq /root/data/downloads/pointnav_gibson_v2.zip

# Repo
cd /root/data && git clone https://github.com/LIJianxuanLeo/bev-vawa-lite.git
cd bev-vawa-lite
sed -i 's/requires-python = ">=3.10"/requires-python = ">=3.9"/' pyproject.toml
HP=/root/miniconda3/envs/habitat/bin
$HP/pip install -e . --no-deps
export PYTHONPATH=$PWD

# 72 scenes × 8 ep/scene × 16 samples/ep ≈ 9 000 train samples; ~5 min.
$HP/python scripts/generate_data_habitat.py \
    --config configs/habitat/gibson.yaml \
    --dataset gibson_v2 \
    --scene-dir /root/data/scene_datasets/gibson \
    --episode-dir /root/data/datasets/pointnav/gibson/v2 \
    --split train \
    --out /root/data/gibson_pointnav_v2_shards/train \
    --max-episodes-per-scene 8

$HP/python scripts/generate_data_habitat.py \
    --config configs/habitat/gibson.yaml \
    --dataset gibson_v2 \
    --scene-dir /root/data/scene_datasets/gibson \
    --episode-dir /root/data/datasets/pointnav/gibson/v2 \
    --split val \
    --out /root/data/gibson_pointnav_v2_shards/val \
    --max-episodes-per-scene 4

# Three-stage training. With the §4 patches, each stage is 30–60 s.
$HP/python scripts/train.py --config configs/habitat/gibson.yaml \
    --stage a --data /root/data/gibson_pointnav_v2_shards/train \
    --out /root/data/runs/gibson --epochs 10

$HP/python scripts/train.py --config configs/habitat/gibson.yaml \
    --stage b --data /root/data/gibson_pointnav_v2_shards/train \
    --out /root/data/runs/gibson \
    --in-ckpt /root/data/runs/gibson/stage_a.pt --epochs 10

$HP/python scripts/train.py --config configs/habitat/gibson.yaml \
    --stage c --data /root/data/gibson_pointnav_v2_shards/train \
    --out /root/data/runs/gibson \
    --in-ckpt /root/data/runs/gibson/stage_b.pt --epochs 5

# Evaluation — note --scenes takes a SPACE-SEPARATED LIST OF .glb FILES,
# not a directory. Build it from the val episode pack.
VAL_SCENES=()
for f in /root/data/datasets/pointnav/gibson/v2/val/content/*.json.gz; do
    name=$(basename "$f" .json.gz)
    glb="/root/data/scene_datasets/gibson/${name}.glb"
    [ -f "$glb" ] && VAL_SCENES+=("$glb")
done

for SEED in 12345 42 7 31337; do
    $HP/python scripts/eval_habitat.py \
        --config configs/habitat/gibson.yaml \
        --scenes "${VAL_SCENES[@]}" \
        --policy bev_vawa --ckpt /root/data/runs/gibson/stage_c.pt \
        --n-episodes 100 --seed $SEED \
        --method-name "BEV-VAWA Gibson seed=$SEED" \
        --results /root/data/runs/gibson/main_table_habitat.csv
done
```

---

## 7. Observed training curves and known-good losses

With `--epochs 10/10/5` after the §4 patches:

```
Stage A (VA head):
  epoch 0  loss 0.938
  epoch 9  loss 0.587

Stage B (WA head with L_dyn + L_deadend):
  epoch 0  loss 3.543
  epoch 9  loss 1.951

Stage C (joint fine-tune, λ_dyn, λ_deadend scaled by 0.4):
  epoch 0  loss 1.794
  epoch 4  loss 1.753
```

On training-set observations the model reaches **>80% top-1 best_k
accuracy** (see `scripts/` — not yet wired into a public tool; inline
diagnostic in the main report). Runtime closed-loop SR on `val` is
**0.00–0.01 without the reactive-safety wrapper**, which matches the
≤0.07 SR of the perception-free `straight` baseline — i.e. pure imitation
from teleport-expert shards does not transfer to closed-loop 3D scanned
rooms without either (a) DAGger-style closed-loop data aggregation or
(b) the reactive safety wrapper tuned for Habitat scene statistics.
See the main report, §10.2 and §9.3.

---

## 8. Cost log — first clean run on ebcloud `bob-eci.4090-slim.5large`

| Phase | Wall time | Billable @ ¥1.89/h |
|---|---|---|
| Scene download (1.4 GiB, single-thread wget) | 8 min | ¥0.25 |
| Episode download + unzip | 1 min | ¥0.03 |
| habitat-sim env bootstrap | 8 min | ¥0.25 |
| Data-gen train + val | 5 min | ¥0.16 |
| 3-stage training (10/10/5 epochs, with §4 patches) | 2 min | ¥0.06 |
| Eval 4 seeds × 100 episodes | 7 min | ¥0.22 |
| **Net minimum, from-scratch reproduction** | **~31 min** | **¥0.98** |

Adding a round of `--safety` ablation (a second 4-seed eval pass) is
an additional ~7 min / ¥0.22.

Persistent-storage cost on the 256 GiB `gibdata` PVC is **¥0.04/h**
(standard ECFS), i.e. ~¥0.96/day independent of whether the compute is
on or off.

---

## 9. Failure modes — what NOT to debug first

These are things we ruled out and you should too, in this order:

1. **"The model is broken."** Run the offline diagnostic in
   `scripts/` / a one-off Python (load stage_c.pt, feed it a shard
   sample, print `select_waypoint` output) — if `best_k == expert_best_k`
   for >80% of samples, the model is fine and the problem is in the
   closed-loop / env path.
2. **"tmpfs will save us."** No. The shard cache is the bottleneck.
   Applying the §4 patches without moving anything to `/dev/shm` is
   sufficient.
3. **"The scene data is corrupt."** The `SSD Load Failure!` warnings on
   every scene load are benign — they refer to missing `info_semantic.json`
   files, which we do not use (we only render depth). Filter them out of
   logs and they stop being alarming.
4. **"We need the full 10 GiB trainval pack."** Not for PointNav v2, no.
   All 72 train + 14 val episode scenes are in the 1.4 GiB 4+ pack.

---

## 10. Full trainval pack (optional, only if you actually need the
    extra "2+ quality" scenes)

```
https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip
# 10.1 GiB, single-thread wget ~30 min on Chinese mainland cloud.
```

Do NOT try to accelerate with `aria2c -x16`; see §2.1.
