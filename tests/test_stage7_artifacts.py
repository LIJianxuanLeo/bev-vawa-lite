"""Stage 7 gate: figures + paper skeleton."""
from __future__ import annotations
from pathlib import Path
import pytest

from scripts.make_figures import main as make_figures


def test_paper_skeleton_has_sections():
    p = Path("paper/paper.md")
    assert p.exists()
    text = p.read_text()
    for section in ["## Abstract", "## 1. Introduction", "## 2. Related Work",
                     "## 3. Method", "## 4. Training", "## 5. Experimental Setup",
                     "## 6. Results", "## 7. Conclusion"]:
        assert section in text, f"missing section: {section}"


def test_figures_pipeline(tmp_path):
    out = make_figures(tmp_path, dummy=True)
    for key in ("main", "ablation", "latency", "arch"):
        p = out[key]
        assert p.exists()
        assert p.stat().st_size > 500, f"{key} figure looks empty"


def test_figures_from_real_csv(tmp_path):
    # when a caller already has a CSV, the pipeline should respect it
    (tmp_path / "main_table.csv").write_text(
        "method,SR,SPL,CollisionRate,PathLenRatio,LatencyMs\n"
        "A,0.1,0.05,0.5,1.5,5\nB,0.9,0.8,0.05,1.05,10\n"
    )
    (tmp_path / "ablation_table.csv").write_text(
        "ablation,SR,SPL,CollisionRate\n"
        "Full,0.8,0.7,0.1\nNo-WA,0.6,0.5,0.2\n"
    )
    out = make_figures(tmp_path, dummy=False)
    assert out["main"].exists()
