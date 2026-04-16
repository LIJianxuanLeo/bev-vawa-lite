"""Tiny YAML config loader with dotted-key overrides."""
from __future__ import annotations
from pathlib import Path
import copy
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # optional `inherit:` key pointing at another yaml
    if "inherit" in cfg:
        parent = load_config(path.parent / cfg.pop("inherit"))
        cfg = _deep_merge(parent, cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
