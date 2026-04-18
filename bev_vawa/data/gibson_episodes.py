"""Gibson PointNav v2 episode loader.

Parses Habitat's ``pointnav_gibson_v2`` episode pack. The pack ships as
``.json.gz`` files either flat (e.g. ``train/train.json.gz``) or under a
``content/`` subdirectory grouping episodes by scene; both layouts are
handled.

Each episode is yielded as a plain dict with the fields the offline data
rollout needs:

    {
      'episode_id':        str,
      'scene_id':          str,                 # e.g. 'gibson/Allensville.glb'
      'start_position':    np.ndarray (3,),
      'start_rotation':    np.ndarray (4,) or None,
      'goal_position':     np.ndarray (3,),
      'geodesic_distance': float,
      'info':              dict                 # raw episode info, if any
    }

Reference: bev_vawa_lite_gibson_pointnav_v2_training_plan_zh.md §1.1, §5.2.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional
import numpy as np


def _load_json_gz(path: Path) -> dict:
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def _episode_files(episode_dir: Path, split: str) -> list[Path]:
    """Return all ``.json.gz`` files for a split, handling both layouts."""
    base = episode_dir / split
    if not base.exists():
        # also tolerate episode_dir pointing directly at a split
        base = episode_dir
    content = base / "content"
    files: list[Path] = []
    if content.is_dir():
        files.extend(sorted(content.glob("*.json.gz")))
    files.extend(sorted(base.glob("*.json.gz")))
    # de-dup (some packs have both top-level and content/)
    seen = set()
    dedup = []
    for f in files:
        key = f.name
        if key in seen:
            continue
        seen.add(key)
        dedup.append(f)
    return dedup


def iter_episodes(
    episode_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
    scene_filter: Optional[Iterable[str]] = None,
) -> Iterator[dict]:
    """Yield episode dicts from a Gibson PointNav v2 pack.

    Parameters
    ----------
    episode_dir
        Root directory, e.g. ``data/datasets/pointnav/gibson/v2``.
    split
        One of ``train``, ``val``, ``test``.
    limit
        Cap on number of episodes yielded. ``None`` = unlimited.
    scene_filter
        Optional iterable of scene basenames (``Allensville``). Only episodes
        whose ``scene_id`` contains one of these stems are yielded.
    """
    root = Path(episode_dir)
    files = _episode_files(root, split)
    if not files:
        raise FileNotFoundError(
            f"no .json.gz files under {root/split} (also tried {root/'content'})"
        )
    scene_filter_set = None
    if scene_filter is not None:
        scene_filter_set = {s.lower() for s in scene_filter}

    n = 0
    for f in files:
        data = _load_json_gz(f)
        episodes = data.get("episodes", [])
        for ep in episodes:
            scene_id = str(ep.get("scene_id", ""))
            if scene_filter_set is not None:
                stem = Path(scene_id).stem.lower()
                if stem not in scene_filter_set:
                    continue
            goals = ep.get("goals", [])
            if not goals:
                continue
            goal_pos = np.asarray(goals[0].get("position", [0.0, 0.0, 0.0]),
                                  dtype=np.float32)
            start_pos = np.asarray(ep.get("start_position", [0.0, 0.0, 0.0]),
                                   dtype=np.float32)
            start_rot = ep.get("start_rotation", None)
            if start_rot is not None:
                start_rot = np.asarray(start_rot, dtype=np.float32)
            yield {
                "episode_id": str(ep.get("episode_id", f"{f.stem}_{n}")),
                "scene_id": scene_id,
                "start_position": start_pos,
                "start_rotation": start_rot,
                "goal_position": goal_pos,
                "geodesic_distance": float(ep.get("info", {}).get("geodesic_distance",
                                                                   ep.get("geodesic_distance", 0.0))),
                "info": ep.get("info", {}),
            }
            n += 1
            if limit is not None and n >= limit:
                return


def resolve_scene_glb(scene_id: str, scene_dir: str) -> str:
    """Resolve a Habitat scene_id (e.g. 'gibson/Allensville.glb') to a local
    ``.glb`` path under ``scene_dir``. Falls back to ``scene_dir/<basename>``
    so episode packs that store only the stem also work."""
    p = Path(scene_id)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = Path(scene_dir) / p.name
    if cand.exists():
        return str(cand)
    # also try removing the leading 'gibson/' prefix if present
    if len(p.parts) > 1:
        cand2 = Path(scene_dir) / Path(*p.parts[1:])
        if cand2.exists():
            return str(cand2)
    # final fallback: stem.glb
    cand3 = Path(scene_dir) / (p.stem + ".glb")
    if cand3.exists():
        return str(cand3)
    raise FileNotFoundError(
        f"could not resolve Gibson scene '{scene_id}' under '{scene_dir}'"
    )
