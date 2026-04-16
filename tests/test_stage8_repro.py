"""Stage 8 gate: top-level artifacts exist and scripts are syntactically importable."""
from __future__ import annotations
from pathlib import Path
import importlib
import pytest


def test_readme_exists():
    p = Path("README.md")
    assert p.exists()
    text = p.read_text()
    for k in ("BEV-VAWA-Lite", "Quickstart", "Stage Gates"):
        assert k in text


def test_repro_script_exists_and_executable():
    p = Path("repro.sh")
    assert p.exists()
    assert p.stat().st_mode & 0o100  # user-executable


def test_gitignore_blocks_heavy_outputs():
    gi = Path(".gitignore").read_text()
    for pattern in ("data/", "runs/", "*.pt", "*.npz"):
        assert pattern in gi, f"gitignore missing {pattern}"


def test_scripts_importable():
    import importlib.util, pathlib
    for name in ("generate_data", "train", "eval", "make_figures"):
        path = pathlib.Path("scripts") / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        # don't actually run main(), just compile/import
        spec.loader.exec_module(mod)  # noqa


def test_pyproject_pins_torch():
    import re
    text = Path("pyproject.toml").read_text()
    assert re.search(r"torch\s*>=\s*2\.4", text)
