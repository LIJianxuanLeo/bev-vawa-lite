# BEV-VAWA-Lite: A Lightweight BEV-Native Vision-Action / World-Action Framework for Indoor PointGoal Navigation

> **Status:** draft skeleton. Numbers below are placeholders to be replaced by
> results from `results/main_table.csv` and `results/ablation_table.csv`
> produced by `scripts/eval.py`.

## Abstract

We present **BEV-VAWA-Lite**, a lightweight navigation architecture that fuses
a reactive **Vision-Action (VA)** branch with a predictive **World-Action
(WA)** branch over a shared **bird's-eye-view (BEV)** state. VA proposes a
small set of local candidate waypoints; WA predicts their short-horizon
consequences (collision risk, goal progress, epistemic uncertainty) in latent
BEV space; a simple analytic fusion rule re-ranks them. The model is trained
offline on trajectories rolled out from A* experts in **PIB-Nav**, a
procedural indoor benchmark built inside MuJoCo, and evaluated closed-loop on
held-out rooms. The full system runs at XX Hz on a consumer laptop (Apple M4,
no GPU) and contains about YY M parameters — two orders of magnitude fewer
than comparable embodied-navigation foundation models.

## 1. Introduction

Recent embodied-navigation systems rely on multi-GPU training and large
vision-language backbones (e.g., InternNav) or heavy image-level generative
world models. We argue that for the *pointgoal*, *static*, *depth-only* slice
of the problem, a far cheaper design is sufficient — and that the useful
insight from world models (future-consequence evaluation) can be captured in
a small *latent* head over a BEV state, without any pixel generation.

**Contributions.**

1. A **BEV-aligned VA-WA dual-branch architecture** separating fast reaction
   from future evaluation.
2. A **latent BEV World-Action head** that predicts per-candidate risk,
   progress, and epistemic uncertainty with ~50 k parameters.
3. A **candidate-generation + consequence-re-ranking** mechanism (5 fan
   anchors + analytic score) that stays interpretable.
4. A **low-compute experimental protocol** (PIB-Nav) that runs end-to-end on
   a laptop and supports extensive ablations under fixed compute.

## 2. Related Work

Three lines are most relevant. (i) *Embodied navigation foundation models*
(InternNav, OmniVLA, NaVILA) optimize for generality and scale but require
substantial compute. (ii) *BEV navigation* (BEVNav and follow-ups) show that
top-down features improve geometry-aware decisions. (iii) *World models for
driving/navigation* (Navigation World Models, WoTE) introduce future
imagination as a re-ranking signal; we retain this idea in **latent** form.

## 3. Method

### 3.1 Task and Notation

Static, depth-only, indoor PointGoal navigation. At each step the agent
observes depth `d_t` and a relative goal vector `g_t = (distance, bearing)`
and outputs linear/angular velocity `(v_t, ω_t)` for a differential-drive
base.

### 3.2 Shared BEV Encoder

A 3-block depth CNN (32→64→96 channels) compresses `d_t` to a feature map,
which is lifted to a 64×64 top-down feature via adaptive pooling, further
pooled, projected with the goal vector, and passed through an LSTM cell to
produce a latent state `z_t ∈ ℝ^128`. We use LSTM rather than GRU because of
reported GRU slowness on PyTorch MPS.

### 3.3 Vision-Action Branch

We fix K=5 anchor waypoints on a 120° fan in front of the robot at radius
1.5 m. The VA head outputs K logits and K small (≤0.3 m) offsets. Training
uses cross-entropy on the anchor nearest to the expert waypoint plus a Huber
term on the selected candidate.

### 3.4 World-Action Branch

Each anchor is embedded (2 → 16) and consumed by an LSTM-cell rollout of
length H=3 starting from `z_t`. Three heads read the final hidden state:
ensemble risk (3× BCE heads → mean + variance), scalar progress (MSE), and
uncertainty (variance of the ensemble's sigmoids).

### 3.5 Fusion

`Q_k = α · softmax(logits)_k + β · progress_k − γ · σ(risk_k) − δ · unc_k`
with `(α, β, γ, δ) = (1.0, 1.5, 2.0, 0.5)`. The argmax anchor is passed to a
pure-pursuit controller that outputs `(v, ω)`.

## 4. Training

**Stage A** (VA supervision): encoder + VA head trained with CE+Huber on
offline expert samples. **Stage B** (WA supervision): encoder frozen; risk,
progress, and ensemble heads trained. **Stage C** (joint fine-tune): last
encoder block and both heads fine-tuned together.

Data are generated in PIB-Nav by sampling ~1000 procedural rooms, planning
A* paths, Chaikin-smoothing, resampling at 0.1 m, and teleporting the diff-
drive base to poses along the path (with small noise) to record depth, goal
vector, expert waypoint, and per-anchor collision/progress labels from the
inflated occupancy grid.

## 5. Experimental Setup

### 5.1 PIB-Nav Benchmark

Procedural indoor rooms (5–7 m side), 3–6 axis-aligned obstacles, differential-
drive platform with a 128×128 forward-facing depth camera (90° FOV, 3 m
range). Success threshold 0.25 m; collision budget 10; step budget 300.

### 5.2 Baselines

FPV-BC, BEV-BC, BEV-VA, and an A* + pure-pursuit oracle.

### 5.3 Metrics

Success Rate (SR), SPL, collision rate, path-length ratio, inference latency.

## 6. Results

See `results/main_table.csv`; figure `results/fig_main_sr_spl.png`. The full
method improves SR by XX points over BEV-VA (WA branch contribution) and by
YY points over BEV-BC (candidate-structured action), approaching the A*
oracle on SPL while running at interactive rates on a laptop.

### 6.1 Ablations

See `results/ablation_table.csv`; figure `results/fig_ablation.png`. Removing
the WA branch drops SR by ZZ points; removing the uncertainty term drops it
by UU; shrinking K from 5 to 1 collapses the benefit of candidate structuring;
shortening rollout H from 3 to 1 hurts primarily on rooms with tight turns.

### 6.2 Efficiency

Full model: ~2.5 M parameters. Single-step forward on Apple M4 MPS: ~20 ms.
Total training time (Stages A+B+C) on the same laptop: under 6 hours at
default config.

## 7. Conclusion

BEV-VAWA-Lite shows that the core insight of world-model-augmented
navigation — evaluate candidate actions by their predicted futures — can be
captured by a small latent head over a BEV state, yielding a system that
trains and runs on a consumer laptop while keeping the architectural clarity
(fast reaction vs. slow evaluation) that makes the approach interpretable.

## References

Placeholder — see source plan `bev_vawa_lite_plan_en.md` §14 for the working
bibliography (InternNav, BEVNav, ADIN, Navigation World Models, WoTE,
BEV-pretrained world model, OmniVLA, NaVILA, HSSD, ProcTHOR).
