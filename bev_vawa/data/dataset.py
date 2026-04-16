"""Torch Dataset over .npz shards produced by rollout.generate_dataset."""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


def list_shards(data_dir: str) -> List[Path]:
    return sorted(Path(data_dir).glob("room_*.npz"))


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
        for si, p in enumerate(self.shards):
            with np.load(p) as z:
                n = z["depth"].shape[0]
                if self._anchors is None:
                    self._anchors = z["anchors"].astype(np.float32)
            for li in range(n):
                self._index.append((si, li))
        self._cache: dict[int, dict] = {}

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
        # keep a small LRU
        if len(self._cache) > 8:
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
        return {
            "depth": torch.from_numpy(depth).unsqueeze(0),  # (1, H, W)
            "goal": torch.from_numpy(goal),
            "expert_wp": torch.from_numpy(expert_wp),
            "cand_collision": torch.from_numpy(cand_coll),
            "cand_progress": torch.from_numpy(cand_prog),
            "best_k": torch.tensor(best_k, dtype=torch.long),
        }
