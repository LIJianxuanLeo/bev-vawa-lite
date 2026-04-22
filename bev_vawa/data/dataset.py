"""Torch Dataset over .npz shards produced by rollout.generate_dataset."""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


def list_shards(data_dir: str) -> List[Path]:
    """List shards produced by any of the rollout sources:

    - ``room_*.npz``   — MuJoCo / PIB-Nav rollout
    - ``scene_*.npz``  — Habitat / Gibson teleport-expert rollout
    - ``dagger_*.npz`` — DAGger closed-loop aggregation (see
      ``scripts/dagger_aggregate_habitat.py``)

    Also follows symlinks, so a "mix" directory that symlinks shards from
    several sources will load them all in one ``NavShardDataset``.
    """
    d = Path(data_dir)
    return sorted(
        list(d.glob("room_*.npz"))
        + list(d.glob("scene_*.npz"))
        + list(d.glob("dagger_*.npz"))
    )


class NavShardDataset(Dataset):
    """Flatten per-room shards into a single sample-level dataset.

    Loads all shard metadata at construction time but memory-maps arrays lazily.
    Each shard holds many samples; we index into them by (shard_id, local_idx).
    """

    def __init__(self, data_dir: str, depth_max: float = 3.0):
        self.shards = list_shards(data_dir)
        if not self.shards:
            raise FileNotFoundError(f"No .npz shards in {data_dir}")
        self.depth_max = float(depth_max)
        self._index: list[tuple[int, int]] = []
        self._anchors = None
        # Schema detection:
        #   v1 — PIB-Nav / MuJoCo; no schema_version, no v2/v3 keys
        #   v2 — Gibson teleport; schema_version=2, has future_depth +
        #        cand_deadend
        #   v3 — HM3D / DAGger with semantic; schema_version=3, has v2
        #        keys + semantic (H, W) int map and/or future_semantic
        #        (H_future, Hi, Wi) int map
        # Per-shard flags let __getitem__ emit extra tensors without
        # breaking older datasets.
        self._schema_v2: list[bool] = []
        self._schema_v3: list[bool] = []
        for si, p in enumerate(self.shards):
            with np.load(p) as z:
                n = z["depth"].shape[0]
                if self._anchors is None:
                    self._anchors = z["anchors"].astype(np.float32)
                ver = int(z["schema_version"].item()) if "schema_version" in z.files else 1
                v2 = (
                    ver >= 2
                    and "future_depth" in z.files
                    and "cand_deadend" in z.files
                )
                v3 = v2 and ver >= 3 and "semantic" in z.files
            self._schema_v2.append(bool(v2))
            self._schema_v3.append(bool(v3))
            for li in range(n):
                self._index.append((si, li))
        self._cache: dict[int, dict] = {}
        # Aggregate flags — the trainer uses these to decide whether it
        # can train the corresponding losses.
        self.has_future = all(self._schema_v2)
        self.has_deadend = all(self._schema_v2)
        self.has_semantic = all(self._schema_v3)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def anchors(self) -> np.ndarray:
        assert self._anchors is not None
        return self._anchors

    def _load_shard(self, si: int) -> dict:
        if si in self._cache:
            return self._cache[si]
        with np.load(self.shards[si]) as z:
            shard = {k: z[k] for k in z.files}
        # Cache size 128 covers the Gibson Habitat 4+ subset (~86 shards)
        # without thrashing during closed-loop DataLoader iteration. Profiling
        # showed an 80x speedup (5.5 min/epoch -> 3.7 sec/epoch) over the
        # original cache=8 when combined with persistent_workers=True (see
        # stage_{a,b,c}.py) and num_workers=8 (see configs/habitat/gibson.yaml).
        # Each worker holds ~1.6 GB resident when full; at num_workers=8 that
        # is ~13 GB RAM, well within the 32 GB budget of the Habitat train
        # node. Documented in docs/gibson_remote_run.md.
        if len(self._cache) > 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[si] = shard
        return shard

    def __getitem__(self, i: int) -> dict:
        si, li = self._index[i]
        shard = self._load_shard(si)
        depth = shard["depth"][li].astype(np.float32) / self.depth_max  # normalize to [0, 1]
        goal = shard["goal"][li].astype(np.float32)
        expert_wp = shard["expert_wp"][li].astype(np.float32)
        cand_coll = shard["cand_collision"][li].astype(np.float32)
        cand_prog = shard["cand_progress"][li].astype(np.float32)
        best_k = int(shard["best_k"][li])
        sample = {
            "depth": torch.from_numpy(depth).unsqueeze(0),  # (1, H, W)
            "goal": torch.from_numpy(goal),
            "expert_wp": torch.from_numpy(expert_wp),
            "cand_collision": torch.from_numpy(cand_coll),
            "cand_progress": torch.from_numpy(cand_prog),
            "best_k": torch.tensor(best_k, dtype=torch.long),
        }
        # v2 extras — only present on Gibson shards (schema_version >= 2).
        if self._schema_v2[si]:
            fd = shard["future_depth"][li].astype(np.float32) / self.depth_max   # (H, H_im, W_im)
            # shape to (H, 1, H_im, W_im) so the encoder can treat each future
            # step as a (B, 1, H_im, W_im) batch when needed.
            sample["future_depth"] = torch.from_numpy(fd).unsqueeze(1)
            if "future_goal" in shard:
                sample["future_goal"] = torch.from_numpy(
                    shard["future_goal"][li].astype(np.float32)
                )                                                                 # (H, 2)
            sample["cand_deadend"] = torch.from_numpy(
                shard["cand_deadend"][li].astype(np.float32)
            )                                                                     # (K,)
        # v3 extras — semantic label maps, only on HM3D / DAGger shards with
        # schema_version >= 3. Stored as int8 class labels on disk (not
        # one-hot) to keep shard size manageable; the encoder one-hots them
        # on-the-fly.
        if self._schema_v3[si]:
            sem = shard["semantic"][li].astype(np.int64)                         # (H, W)
            sample["semantic"] = torch.from_numpy(sem)
            if "future_semantic" in shard:
                fsem = shard["future_semantic"][li].astype(np.int64)             # (H_future, H, W)
                sample["future_semantic"] = torch.from_numpy(fsem)
        return sample
