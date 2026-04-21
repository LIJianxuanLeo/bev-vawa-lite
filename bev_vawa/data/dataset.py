"""Torch Dataset over .npz shards produced by rollout.generate_dataset."""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


def list_shards(data_dir: str) -> List[Path]:
    """List shards produced by either the MuJoCo rollout (``room_*.npz``) or
    the Habitat rollout (``scene_*.npz``)."""
    d = Path(data_dir)
    return sorted(list(d.glob("room_*.npz")) + list(d.glob("scene_*.npz")))


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
        # Schema detection: v1 shards lack 'schema_version' and the v2-only
        # keys. We record the per-shard flag so __getitem__ can expose extra
        # tensors without breaking v1 datasets.
        self._schema_v2: list[bool] = []
        for si, p in enumerate(self.shards):
            with np.load(p) as z:
                n = z["depth"].shape[0]
                if self._anchors is None:
                    self._anchors = z["anchors"].astype(np.float32)
                v2 = (
                    "schema_version" in z.files
                    and int(z["schema_version"].item()) >= 2
                    and "future_depth" in z.files
                    and "cand_deadend" in z.files
                )
            self._schema_v2.append(bool(v2))
            for li in range(n):
                self._index.append((si, li))
        self._cache: dict[int, dict] = {}
        # Aggregate flag — the trainer uses this to decide whether it can
        # train the dynamics / dead-end losses.
        self.has_future = all(self._schema_v2)
        self.has_deadend = all(self._schema_v2)

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
        return sample
