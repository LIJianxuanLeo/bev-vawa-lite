# CLAUDE.md — Project Operating Guide

This file is read automatically by Claude at the start of every session.
It encodes the **document policy** for the BEV-VAWA-Lite paper project so
that we (Claude + the author) stay consistent across sessions.

---

## 1. Documents under `paper/` — lifecycle and roles

The `paper/` directory holds several artefacts. Each one has a distinct
role and a distinct update policy. **Respect the table below exactly**;
do not edit a "frozen" document unless the author explicitly asks.

| File | Role | Update policy | Language |
|---|---|---|---|
| `paper/full_report.tex` + `.pdf` | **Technical exposition** of the method — architecture, algorithms, data pipeline, engineering notes. Treated as a living design document. | **Active. Keep updating in real time** as experiments + code evolve. Bilingual EN + 中文 per the existing `\bisec{...}{...}` / `\en{...}` / `\zh{...}` macros. | Bilingual (EN + 中文) |
| `paper/stage_report.tex` + `.pdf` + `.html` | Milestone progress report from earlier in the project. Snapshot of state at commit `1a500a1`. | **FROZEN. Do not modify.** If something changes in the project that would contradict stage\_report, acknowledge it in `full_report.tex` or `journal.tex` instead. | English |
| `paper/paper.md` | Earlier markdown draft that pre-dated `full_report.tex`. | **FROZEN. Do not modify.** Superseded by `full_report.tex` for the technical-route role and by the forthcoming `journal.tex` for the publication role. | English |
| `paper/journal.tex` → `journal.pdf` | **The final publication paper.** See §2 for full spec. | **Created only after all experiments are complete.** No partial drafts. | English only |
| `paper/presentation.pptx` | 10-minute academic talk slides. See §3. | **Created only after `journal.tex` is finalised.** | English |

### Operating rule

- When the author says "update the paper", ask **"`full_report` or `journal`?"** unless it is obvious which is meant.
- When the author mentions a finding, metric, ablation, or figure:
  - Update `full_report.tex` immediately (it is the living record).
  - **Do not** touch `stage_report.*` or `paper.md` — they are historical artefacts.
- If the author asks to fix something in `stage_report.*` or `paper.md`, confirm first: "These are frozen. Do you want me to fix it anyway, or port the fix to `full_report.tex`?"

---

## 2. `journal.tex` — final publication paper specification

To be produced **after all experiments are finished**, not before.

### 2.1 Formatting (hard requirements)

| Property | Value |
|---|---|
| Paper size | A4 |
| Margins | **1 inch** on all sides |
| Main font | **Times New Roman** |
| Body font size | **12 pt** |
| Line spacing | **1.5 ×** |
| Text colour | **Pure black only** (no coloured text, no coloured captions) |
| Math font | Default LaTeX math (Computer Modern Math is acceptable; Times-alike math is preferred if easy) |
| Language | **English only** (no 中文, no bilingual passages — this is the opposite of `full_report.tex`) |

LaTeX preamble skeleton to use:

```latex
\documentclass[12pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{setspace}
\onehalfspacing                  % 1.5x line spacing
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs,array,graphicx,float,tabularx}
\usepackage{algorithm,algpseudocode}
\usepackage[hidelinks]{hyperref}
% IEEE-style numeric citations
\usepackage[numbers,sort&compress]{natbib}
\bibliographystyle{IEEEtran}
```

Compile with `tectonic -X compile journal.tex`. `tectonic` auto-downloads
`IEEEtran.bst` on first build.

### 2.2 Citation style — IEEE numeric (engineering / technical)

- **In-text**: square-bracketed numbers, e.g. "… as shown in [1], [2]–[4], [7]."
- **Reference list**: IEEE format, one entry per line, numbered in order of first appearance. Produced by `\bibliographystyle{IEEEtran}` + a `.bib` file, or by hand-written `thebibliography` environment matching IEEE punctuation exactly.

Reference entry examples (IEEE style):

```
[1] P. Anderson, A. Chang, D. S. Chaplot et al., "On evaluation of embodied
    navigation agents," arXiv:1807.06757, Jul. 2018.
[2] D. Ha and J. Schmidhuber, "World models," in Proc. Adv. Neural Inf.
    Process. Syst. (NeurIPS), 2018, pp. 2455–2467.
[3] J. Philion and S. Fidler, "Lift, splat, shoot: Encoding images from
    arbitrary camera rigs by implicitly unprojecting to 3D," in Proc.
    Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 194–210.
```

Key IEEE conventions to preserve:

- Author list: "F. Last" initials-first.
- **Use "et al." when ≥ 4 authors in the list (many IEEE journals)**, or list all (some IEEE conferences). Follow the first style by default.
- Journal / conference names abbreviated ("*Proc.*", "*IEEE Trans. Robot.*", etc.).
- Article title in sentence case, in double quotes.
- Journal / conference title in italics.
- Volume / issue / pages / year at the end.
- arXiv entries: `arXiv:YYMM.NNNNN, Month Year.`

### 2.3 Content checklist (must all be present)

- Title, authors (placeholder), affiliations
- Abstract (≤ 250 words)
- **Introduction** — motivation, contributions bullet list (3–5 items), paper organisation
- **Related Work** — grouped by theme, each citation cross-referenced numerically
- **Problem Formulation** — formal task definition, notation, metrics (SR, SPL, Coll., PLR, Latency)
- **Method** — the full architecture, broken into clear subsections:
  - Shared BEV Encoder (mathematical definition of the geometry lift, including the occupancy / free-space / goal-prior channels as equations)
  - Vision-Action (VA) Head — multi-waypoint anchor design, Best-K supervision with Huber offset, loss equations
  - World-Action (WA) Head — latent rollout equations, risk / progress / uncertainty heads, `L_dyn`, dead-end head
  - Analytic Fusion — `Q_k = αs_k + βp_k − γr_k − δu_k − η·d_k` derivation and interpretation
  - Reactive Safety Layer — v3 forward APF + side scrape guard equations
  - Pure-Pursuit Controller — equations
- **Algorithms** — at least two `algorithm` environments with `algpseudocode`:
  1. **Algorithm 1**: three-stage training curriculum (Stage A / B / C)
  2. **Algorithm 2**: closed-loop inference loop with WA fusion and safety wrapping
  Optionally a third for `L_dyn` + dead-end sample preparation.
- **Experimental Setup** — benchmarks (PIB-Nav, Gibson Habitat), seeds, splits, hyperparameters in a boxed table
- **Results** — main table, ablation table, cross-seed paired Δ analysis, Habitat supplementary. Every numeric claim must map to a CSV file in `results/`.
- **Discussion** — covariance shift framing, why WA + safety are jointly necessary, scope limits.
- **Conclusion** — 1–2 paragraphs
- **References** (IEEE numeric, Section 2.2)

### 2.4 Figures and tables (must include)

All figures must be **pure black-and-white compatible** (no colour-only encodings) and **pure black text**. Use line style + marker shape + dash pattern for differentiation, not colour.

Required figures (all already present in `results/` as PNG):

- `results/architecture.png` — system block diagram
- `results/fig_main_sr_spl.png` — main SR/SPL comparison with cross-seed error bars
- `results/fig_per_seed_delta.png` — per-seed paired ΔSR for the safety layer
- `results/fig_ablation.png` — 2-panel ablation
- `results/fig_latency_sr.png` — efficiency trade-off

Required tables (data in `results/`):

- Main table — A* / baselines / BEV-VAWA ± safety / cross-seed means
- Ablation table — `no_wa`, `no_unc`, `h1` paired ΔSR
- Seed-stability table — per-seed SR and paired Δ
- Habitat supplementary table — Gibson convergence + closed-loop gap

### 2.5 Pseudocode style

Use `algorithm` + `algpseudocode` packages. Keep to ≤ 20 lines per algorithm.

```latex
\begin{algorithm}[H]
\caption{Three-Stage Curriculum for BEV-VAWA}
\label{alg:curriculum}
\begin{algorithmic}[1]
\Require shards $\mathcal{D}$; encoder $\phi$; VA head $\pi^{\mathrm{VA}}$;
         WA head $\pi^{\mathrm{WA}}$; epochs $E_A, E_B, E_C$
\State \textbf{Stage A}: optimise $\phi \cup \pi^{\mathrm{VA}}$ on $\mathcal{L}_{VA}$ for $E_A$ epochs
\State \textbf{Stage B}: freeze $\phi \cup \pi^{\mathrm{VA}}$; optimise $\pi^{\mathrm{WA}}$ on
       $\mathcal{L}_{\mathrm{risk}} + \mathcal{L}_{\mathrm{prog}} +
       \lambda_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}} +
       \lambda_{\mathrm{dead}}\mathcal{L}_{\mathrm{dead}}$ for $E_B$ epochs
\State \textbf{Stage C}: unfreeze all; joint FT with halved $\lambda$s for $E_C$ epochs
\end{algorithmic}
\end{algorithm}
```

### 2.6 "Experiments complete" trigger — what counts as done

Do not start writing `journal.tex` until **all** of the following exist as
CSV / PNG / checkpoint in `results/` and `runs/`:

1. PIB-Nav cross-seed table for both `full` and `full+safety` — ✅ already have
2. Ablations for `no_wa`, `no_unc`, `h1` — ✅ already have
3. Baselines FPV-BC / BEV-BC / BEV-VA at 4 seeds — ✅ already have
4. Gibson Habitat closed-loop eval at 4 seeds **with `--safety`** — ✅ already have
5. Gibson safety-parameter sweep (4 variants × 4 seeds) — ✅ already have
6. Gibson DAGger-1 paired eval — ✅ already have

**Optional Gibson / HM3D uplift track** (in-repo tooling is ready; results
pending next remote-GPU sessions):

7. **Discrete-action interface** — `configs/habitat/gibson.yaml` now sets
   `env.discrete_actions: true`. Expected Gibson/HM3D SR ~0.10-0.20
   (vs. baseline 0.005). Enabled automatically on next Stage-C eval.
8. **Learned collision head** — `configs/habitat/gibson.yaml` sets
   `wa.enable_coll_head: true`, `wa.lambda_coll_head: 0.3`,
   `fusion.mu: 1.5`. Active on any retraining from scratch.
9. **Semantic-mask perception** — `configs/habitat/hm3d.yaml` is a new
   HM3D-specific config with `env.use_semantic: true` and
   `bev.use_semantic: true`. Requires HM3D v0.2 + pointnav_hm3d_v1
   download (see file header). Expected SR ~+10pp over #7+#8.
10. **DAGger β-schedule (3 rounds)** — `scripts/run_dagger_iteration.sh`
    accepts a second arg `beta`:
      ```
      bash scripts/run_dagger_iteration.sh iter1 0.8
      CKPT_IN=/root/data/runs/gibson_dagger_iter1/stage_c.pt \
          bash scripts/run_dagger_iteration.sh iter2 0.4
      CKPT_IN=/root/data/runs/gibson_dagger_iter2/stage_c.pt \
          bash scripts/run_dagger_iteration.sh iter3 0.0
      ```
    Canonical DAGger remedy for covariance shift; expected
    SR 0.30-0.45 on top of #7+#8+#9.

When 1–6 are all satisfied, you (Claude) may start `journal.tex`. Results
from 7–10 enrich §10 but are not blocking; the paper's Gibson negative
result + mechanistic analysis already stands on 1–6.

---

## 3. `presentation.pptx` — 10-minute talk slides specification

Produced **after `journal.tex` is finalised**, not before.

### 3.1 Visual design

| Property | Value |
|---|---|
| Dominant colours | **Purple** (primary) + **white** |
| Accent colours | **Red** and **yellow** (use sparingly for call-outs / highlights / failure cases) |
| Style | Academic, minimal, uncluttered |
| Font | Sans-serif, body ≥ 20 pt, titles ≥ 28 pt |
| Slide aspect | 16:9 |

### 3.2 Target length and structure (for a 10-minute talk)

Roughly 10–12 content slides (so ~50 seconds per slide):

1. **Title** — paper title, author, affiliation, 1 "hero" graphic
2. **Problem & Motivation** — 1 sentence, 1 failure GIF or still frame
3. **Contributions** — 3 bullets; one sentence each
4. **Method Overview** — `architecture.png` with labelled callouts
5. **BEV + VA + WA** — split visualisation; highlight WA's `L_dyn` + dead-end in contrasting accent
6. **Analytic Fusion + Safety** — equation `Q_k = αs + βp − γr − δu − η·d`; safety wrapper diagram
7. **Training Curriculum** — A/B/C timeline
8. **Main Results (PIB-Nav)** — reproduce `fig_main_sr_spl.png` with error bars
9. **Cross-Seed ΔSR** — reproduce `fig_per_seed_delta.png`; the +0.093 ± 0.013 headline
10. **Ablations** — reproduce `fig_ablation.png`; highlight WA contribution
11. **Gibson Habitat Supplementary** — training converges, closed-loop gap, negative-result sanity check → *motivates* WA+safety framing
12. **Conclusion + Future Work** — 3 bullets + QR to code repo

### 3.3 Generation tooling

Preferred: generate `.pptx` via `python-pptx` (lets us build slides
programmatically and re-run when figures update). Alternative: ask the
author for permission to produce slides as LaTeX beamer instead.

### 3.4 Trigger

Start producing `presentation.pptx` **only** after:

- `journal.tex` compiles cleanly and all figures / tables are final
- The author explicitly requests the slides

---

## 4. Other operational notes

- **Remote GPU work** (ebcloud 4090) requires `export KUBECONFIG="$HOME/.kube/config:$HOME/.kube/configs/train2-eb.yaml"` before every `kubectl` call. See `docs/gibson_remote_run.md` for the full runbook.
- **The `habitat` conda env** in the remote Pod lives at `/root/miniconda3/envs/habitat/`. The `/root/data/` PVC is persistent across restarts (¥0.04/h storage cost); the GPU compute is metered separately at ¥1.89/h.
- **When in doubt about which document to update**, ask the author. Default answer: `full_report.tex`.
- **Do not delete** `stage_report.*` or `paper.md` — they remain in the repo as historical records.
- **Git commit etiquette**: keep the existing style (`feat:` / `fix:` / `docs:` / `chore:` prefix, first line < 72 chars, body explaining the why).
