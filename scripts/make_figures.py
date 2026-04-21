"""Generate the paper's figures from results CSVs.

Produces:
    results/fig_main_sr_spl.png     -- method-vs-SR/SPL bar chart with cross-seed error bars
    results/fig_per_seed_delta.png  -- paired Delta-SR per seed (Table 2 visualization)
    results/fig_ablation.png        -- ablation bar chart + Delta vs Full
    results/fig_latency_sr.png      -- latency vs SR Pareto scatter
    results/architecture.svg        -- architecture diagram

CSV expectations (see results/main_table.csv):

* Single-seed rows (method does not contain the word "seed" or "mean"):
  used as-is.
* Per-seed rows: method ends with " seed=<int>"; these are grouped by
  family (the method string with " seed=..." stripped) and averaged
  with std reported as error bars.
* "mean<N>seeds" rows: informational only, skipped by this script
  (we recompute mean+std from per-seed rows to keep the pipeline
  self-contained).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS = Path("results")


# ------------------------------------------------------------------ CSV I/O
def _load_csv(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


_SEED_RE = re.compile(r"\s+seed=(\d+)$")
_MEAN_RE = re.compile(r"\s+mean\d+seeds$")


def _parse_family(method: str) -> tuple[str, int | None]:
    """Return (family, seed_or_None). Mean rows return seed=-1."""
    if _MEAN_RE.search(method):
        return _MEAN_RE.sub("", method), -1
    m = _SEED_RE.search(method)
    if m:
        return _SEED_RE.sub("", method), int(m.group(1))
    return method, None


def _gather(rows: list[dict]) -> dict[str, dict]:
    """Group rows by family. Returns {family: {'single': row|None, 'seeds': [(seed, row), ...]}}"""
    by_family: dict[str, dict] = defaultdict(lambda: {"single": None, "seeds": []})
    for r in rows:
        fam, seed = _parse_family(r["method"])
        if seed is None:
            by_family[fam]["single"] = r
        elif seed >= 0:
            by_family[fam]["seeds"].append((seed, r))
        # seed == -1 (precomputed mean row) is ignored
    return by_family


def _stats(family: dict, key: str) -> tuple[float, float]:
    """(mean, std). Uses per-seed rows if >=2 seeds available, else falls back
    to the single-seed row (std=0)."""
    seeds = family["seeds"]
    if len(seeds) >= 2:
        xs = np.array([float(r[key]) for _, r in seeds])
        return float(xs.mean()), float(xs.std(ddof=1))
    if family["single"] is not None:
        return float(family["single"][key]), 0.0
    if len(seeds) == 1:
        return float(seeds[0][1][key]), 0.0
    return float("nan"), 0.0


# ------------------------------------------------------------------ figures
# Curated ordering used for the main figure. Families not in this list are
# skipped (e.g. individual per-seed rows would never appear as their own bar).
_MAIN_ORDER = [
    ("A* Upper Bound",              "A* Upper"),
    ("A* + safety",                 "A* + Safety"),
    ("FPV-BC",                      "FPV-BC"),
    ("BEV-BC",                      "BEV-BC"),
    ("BEV-VA",                      "BEV-VA"),
    ("BEV-VAWA (full)",             "BEV-VAWA"),
    ("BEV-VAWA (full+safety)",      "BEV-VAWA + Safety"),
]


def fig_main(results_dir: Path) -> Path:
    by_family = _gather(_load_csv(results_dir / "main_table.csv"))
    labels, sr_m, sr_s, spl_m, spl_s = [], [], [], [], []
    for fam_key, label in _MAIN_ORDER:
        if fam_key not in by_family:
            continue
        m, s = _stats(by_family[fam_key], "SR")
        if np.isnan(m):
            continue
        labels.append(label)
        sr_m.append(m); sr_s.append(s)
        p, q = _stats(by_family[fam_key], "SPL")
        spl_m.append(p); spl_s.append(q)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b1 = ax.bar(x - w / 2, sr_m, w, yerr=sr_s, capsize=3,
                label="SR", color="#4c78a8", edgecolor="black", linewidth=0.5,
                error_kw=dict(ecolor="black", lw=1))
    b2 = ax.bar(x + w / 2, spl_m, w, yerr=spl_s, capsize=3,
                label="SPL", color="#f58518", edgecolor="black", linewidth=0.5,
                error_kw=dict(ecolor="black", lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, max(0.8, max(sr_m) * 1.25))
    ax.set_ylabel("Score")
    ax.set_title("PointGoal Navigation on PIB-Nav (learned rows: mean ± std over 4 seeds)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    # annotate bar values
    for bars, means in [(b1, sr_m), (b2, spl_m)]:
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = results_dir / "fig_main_sr_spl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_per_seed_delta(results_dir: Path) -> Path | None:
    """Paired Delta-SR per seed: shows the safety improvement is sign-stable
    even when absolute SR fluctuates."""
    by_family = _gather(_load_csv(results_dir / "main_table.csv"))
    full = by_family.get("BEV-VAWA (full)")
    saf = by_family.get("BEV-VAWA (full+safety)")
    if full is None or saf is None:
        return None
    # Build dict seed -> (full_sr, safe_sr). The single-seed (no suffix) row
    # corresponds to seed=12345 by convention.
    full_by_seed: dict[int, float] = {}
    if full["single"] is not None:
        full_by_seed[12345] = float(full["single"]["SR"])
    for s, r in full["seeds"]:
        full_by_seed[s] = float(r["SR"])
    saf_by_seed: dict[int, float] = {}
    if saf["single"] is not None:
        saf_by_seed[12345] = float(saf["single"]["SR"])
    for s, r in saf["seeds"]:
        saf_by_seed[s] = float(r["SR"])

    seeds = sorted(set(full_by_seed) & set(saf_by_seed))
    if len(seeds) < 2:
        return None
    full_sr = [full_by_seed[s] for s in seeds]
    saf_sr = [saf_by_seed[s] for s in seeds]
    deltas = [b - a for a, b in zip(full_sr, saf_sr)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw=dict(width_ratios=[1.5, 1.0]))
    # left: per-seed paired bars
    x = np.arange(len(seeds))
    w = 0.38
    ax1.bar(x - w / 2, full_sr, w, label="BEV-VAWA (full)",
            color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w / 2, saf_sr, w, label="+ Safety",
            color="#e45756", edgecolor="black", linewidth=0.5)
    for xi, (a, b, d) in enumerate(zip(full_sr, saf_sr, deltas)):
        ax1.annotate(f"+{d:.2f}",
                     xy=(xi, max(a, b) + 0.01),
                     ha="center", fontsize=8, color="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seeds])
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Per-seed paired SR (PIB-Nav, n=100 episodes)")
    ax1.set_ylim(0, max(saf_sr) * 1.2)
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    # right: delta distribution (stripplot + mean line)
    jitter = np.random.default_rng(0).normal(scale=0.04, size=len(deltas))
    ax2.scatter(jitter, deltas, s=70, color="#54a24b", edgecolor="black", zorder=3)
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    ax2.axhline(mean_d, color="#333", linestyle="--", zorder=2)
    ax2.axhspan(mean_d - std_d, mean_d + std_d, color="#eee", zorder=1)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([])
    ax2.set_ylabel(r"$\Delta$ SR (Safety − Full)")
    ax2.set_title(f"Paired $\\Delta$SR\nmean={mean_d:+.3f}, std={std_d:.3f}")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax2.set_ylim(0, max(deltas) * 1.4)
    fig.tight_layout()
    out = results_dir / "fig_per_seed_delta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_ablation(results_dir: Path) -> Path:
    rows = _load_csv(results_dir / "ablation_table.csv")
    labels = [r["ablation"] for r in rows]
    sr = [float(r["SR"]) for r in rows]
    spl = [float(r["SPL"]) for r in rows]
    full_sr = sr[labels.index("Full")] if "Full" in labels else sr[0]
    deltas = [v - full_sr for v in sr]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    w = 0.38
    ax1.bar(x - w / 2, sr, w, label="SR", color="#54a24b",
            edgecolor="black", linewidth=0.5)
    ax1.bar(x + w / 2, spl, w, label="SPL", color="#b279a2",
            edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=10)
    ax1.set_ylim(0, max(max(sr), max(spl)) * 1.25)
    ax1.set_title("Ablations (seed 12345, n=100)")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    colors = ["#dddddd" if d == 0 else ("#4c78a8" if d > 0 else "#e45756") for d in deltas]
    ax2.barh(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.axvline(0, color="black", linewidth=0.8)
    # expand limits so labels don't collide with y-axis tick labels
    pad = max(0.015, max(abs(min(deltas)), abs(max(deltas))) * 0.3)
    ax2.set_xlim(min(0, min(deltas)) - pad, max(0, max(deltas)) + pad)
    for xi, d in enumerate(deltas):
        # annotation just OUTSIDE the bar tip (always on the away-from-zero side)
        if d > 0:
            ax2.annotate(f"{d:+.2f}", xy=(d, xi), xytext=(4, 0),
                         textcoords="offset points",
                         va="center", ha="left", fontsize=9)
        elif d < 0:
            ax2.annotate(f"{d:+.2f}", xy=(d, xi), xytext=(-4, 0),
                         textcoords="offset points",
                         va="center", ha="right", fontsize=9)
        else:
            ax2.annotate("0", xy=(0, xi), xytext=(4, 0),
                         textcoords="offset points",
                         va="center", ha="left", fontsize=9, color="#666")
    ax2.set_xlabel(r"$\Delta$SR vs. Full (paired, same seed)")
    ax2.set_title("Paired ablation delta")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax2.invert_yaxis()
    fig.tight_layout()
    out = results_dir / "fig_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_latency_sr(results_dir: Path) -> Path:
    by_family = _gather(_load_csv(results_dir / "main_table.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"A* Upper Bound": "#999", "A* + safety": "#555",
               "FPV-BC": "#9c6", "BEV-BC": "#4c78a8", "BEV-VA": "#f58518",
               "BEV-VAWA (full)": "#e45756",
               "BEV-VAWA (full+safety)": "#b54"}
    for fam_key, label in _MAIN_ORDER:
        if fam_key not in by_family:
            continue
        sr, _ = _stats(by_family[fam_key], "SR")
        lat, _ = _stats(by_family[fam_key], "LatencyMs")
        if np.isnan(sr) or np.isnan(lat):
            continue
        ax.scatter(lat, sr, s=70, color=palette.get(fam_key, "#333"), edgecolor="black", zorder=3)
        ax.annotate(label, (lat, sr), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, zorder=4)
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Latency vs SR (PIB-Nav)")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = results_dir / "fig_latency_sr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_architecture(results_dir: Path) -> Path:
    """Minimal ASCII-ish architecture diagram rendered via matplotlib text."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.95, "BEV-VAWA-Lite Architecture", ha="center", fontsize=14, weight="bold",
            transform=ax.transAxes)
    blocks = [
        (0.05, 0.60, 0.18, 0.15, "Depth 128x128\n+ Goal(2)"),
        (0.28, 0.60, 0.18, 0.15, "BEV Encoder\n(CNN + Lift + LSTM)"),
        (0.51, 0.78, 0.18, 0.12, "VA Head\nK logits + offsets"),
        (0.51, 0.42, 0.18, 0.12, "WA Head\nH-step latent rollout\nRisk / Progress / Unc."),
        (0.74, 0.60, 0.18, 0.15, r"Fusion" + "\n" + r"$Q = \alpha s + \beta p - \gamma r - \delta u$"),
        (0.84, 0.18, 0.14, 0.12, "Pure Pursuit\n(v, w)"),
    ]
    for x, y, w, h, text in blocks:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#eef", edgecolor="#336"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
                transform=ax.transAxes)
    out = results_dir / "architecture.svg"
    fig.savefig(out)
    plt.close(fig)
    return out


def main(results_dir: Path = DEFAULT_RESULTS) -> dict:
    paths: dict[str, Path | None] = {
        "main": fig_main(results_dir),
        "per_seed_delta": fig_per_seed_delta(results_dir),
        "ablation": fig_ablation(results_dir),
        "latency": fig_latency_sr(results_dir),
        "arch": fig_architecture(results_dir),
    }
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    out = main(Path(args.results))
    for k, v in out.items():
        print(f"{k}: {v}")
