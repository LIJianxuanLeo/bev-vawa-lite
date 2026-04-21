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
held-out rooms. The full system runs at ~180 Hz on a consumer laptop (Apple M4, MPS only,
no discrete GPU) and contains about 1.12 M parameters — two orders of
magnitude fewer than comparable embodied-navigation foundation models.

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

Four baselines share the evaluation pipeline with our method:

- **A\* + Pure-Pursuit (oracle)**: ideal planner replanning on the ground-
  truth inflated occupancy grid every step; an upper bound on what the
  controller can do given perfect perception.
- **FPV-BC**: front-pixel depth → 3-block CNN → 5-way discrete action +
  offset regression (no BEV, no VA, no WA).
- **BEV-BC**: same head as FPV-BC but fed our BEV encoder's latent `z_t`.
  Isolates the value of the BEV lift alone.
- **BEV-VA**: BEV latent + our VA multi-waypoint head (K=5 anchors, Best-K
  classification + Huber offset). I.e. our full model with the WA branch
  surgically removed. Isolates the value of the WA branch.

All three imitation baselines are trained with the same 20-epoch Stage-A
recipe as the VA head in our model and evaluated at the same 4 seeds.

### 5.3 Metrics

Success Rate (SR), SPL, collision rate, path-length ratio, inference latency.

## 6. Results

Main closed-loop results on PIB-Nav. For the two learned-policy rows we
report **mean ± std over 4 seeds** (12345, 42, 7, 31337), 100 held-out
episodes each; A\* rows are deterministic and reported at a single seed.
See `results/main_table.csv` and `results/fig_main_sr_spl.png`:

| Method                        | SR            | SPL           | Coll          | PLR  | Lat.\,(ms) |
|-------------------------------|---------------|---------------|---------------|------|------------|
| A\* Upper Bound (oracle)      | 0.48          | 0.47          | 0.54          | 0.89 | 2.76       |
| A\* + Safety                  | 0.50          | 0.49          | 0.50          | 0.90 | 2.63       |
| FPV-BC (mean 4 seeds)         | 0.35 ± 0.03   | 0.34 ± 0.03   | 0.69 ± 0.03   | 0.94 | 1.56       |
| BEV-BC (mean 4 seeds)         | 0.33 ± 0.04   | 0.32 ± 0.04   | 0.69 ± 0.05   | 0.92 | 4.42       |
| BEV-VA (mean 4 seeds)         | 0.33 ± 0.03   | 0.32 ± 0.03   | 0.69 ± 0.04   | 0.92 | 4.13       |
| BEV-VAWA (full, mean 4 seeds) | 0.40 ± 0.05   | 0.39 ± 0.05   | 0.65 ± 0.05   | 0.92 | 5.76       |
| **BEV-VAWA (full + Safety)**  | **0.49 ± 0.05** | **0.49 ± 0.05** | **0.57 ± 0.06** | 0.92 | 6.11 |

The three imitation baselines cluster tightly at SR ≈ 0.33–0.35,
essentially indistinguishable from each other at n=100. Moving the
encoder from raw FPV pixels to the shared BEV latent (FPV-BC → BEV-BC)
is neutral; adding the VA multi-waypoint head (BEV-BC → BEV-VA) is also
neutral. It is only when the WA branch contributes risk / progress /
uncertainty scores into the analytic fusion `Q_k` that SR jumps to 0.40
± 0.05, and adding the safety layer lifts it further to 0.49 ± 0.05.
This is the central claim of the paper: the WA branch is the component
doing load-bearing work — the improvement is not attributable to either
the BEV lift or the VA head alone.

Although the absolute per-seed SR of the learned policy fluctuates by
about ±5 pp — an expected consequence of the small 100-episode test set
and the stochasticity of procedural rooms — the **improvement from the
reactive safety wrapper is highly consistent: paired ΔSR = +0.093 ±
0.013 across the four seeds (+8 to +11 pp, never negative).** With
safety enabled the learned policy matches the reactive-safety A\*
baseline on SR and reaches within 1 pp of SPL; both learned+safety and
A\*+safety sit 2–6 pp above the bare A\* controller upper bound because
the safety layer cuts collisions from ≈0.54–0.65 to ≈0.50–0.57. The
seed=12345 checkpoint alone reports SR 0.45 → 0.53 (+0.08); treating
that single-seed number as the headline would overstate the learned
policy's standalone quality — the safety-layer *improvement*, however, is
real at every seed we evaluated. See the separate *Stage Report*
(`paper/stage_report.pdf`) for the detailed safety-layer analysis.

### 6.1 Ablations

Ablations isolate the two non-obvious design choices of the WA branch.
All runs share the Stage-A checkpoint where applicable; numbers are on
100 held-out episodes at **seed 12345** (single-seed — see caveat on
seed variance in the main table above; ablations are interpreted as
**differences within the same seed** so the per-seed offset cancels).
See `results/ablation_table.csv`:

| Ablation                      | SR   | SPL  | Coll | ΔSR vs Full |
|-------------------------------|------|------|------|-------------|
| Full                          | 0.45 | 0.44 | 0.58 | —           |
| *no_wa* — fusion γ=δ=0        | 0.39 | 0.38 | 0.63 | −6 pp       |
| *no_unc* — fusion δ=0         | 0.45 | 0.44 | 0.58 | 0 pp        |
| *h1* — rollout H=3→1 (retrain)| 0.44 | 0.43 | 0.58 | −1 pp       |

Removing the WA branch entirely drops SR by **6 points**, confirming
its end-task contribution. Shortening the rollout to H=1 costs only
**1 point** — the gap is small because the current WA branch has no
supervision on intermediate rollout latents; adding the `L_dyn`
objective on the Gibson v2 track (§Future Work) is designed to extend
this margin. The uncertainty-only ablation (`no_unc`, δ=0) is
**byte-identical to Full at seed 12345**: the ensemble-variance term
is not large enough to change candidate rankings at test time.
Re-weighting or calibrating the ensemble to make δ·û actually
contribute is a follow-up. Numbers for the K-sweep are deferred
pending re-evaluation under the corrected metric.

### 6.2 Efficiency

Full model: **1.12 M** parameters. Single-step forward on Apple M4
MPS: **5.54 ms** (≈180 Hz), well above the 10 Hz control loop.
Total training time (Stages A+B+C on 1000-room dataset, epochs 20/20/8):
about 3 h on the M4; data generation another ~5.5 h.

### 6.3 Seed-stability protocol

Closed-loop SR on a 100-episode test set of procedural rooms has an
empirical per-seed standard deviation of ≈0.05 on this benchmark
(measured on the four seeds listed in §6). We therefore report the two
learned-policy rows in the main table as mean ± std over 4 seeds, and
treat single-seed numbers as indicative rather than headline. For the
ablation table (§6.1) the same seed is reused across all variants so
per-seed offset cancels; only **ΔSR vs. Full at seed 12345** is
interpreted as signal. The safety-layer effect (ΔSR = +0.093 ± 0.013,
σ ≈ 4× smaller than the raw per-seed SR σ) is an explicit example of
this paired design — the improvement is real even though the absolute
numbers move around.

## 7. Conclusion

BEV-VAWA-Lite shows that the core insight of world-model-augmented
navigation — evaluate candidate actions by their predicted futures — can be
captured by a small latent head over a BEV state, yielding a system that
trains and runs on a consumer laptop while keeping the architectural clarity
(fast reaction vs. slow evaluation) that makes the approach interpretable.

## References

1. P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun,
   J. Kosecka, J. Malik, R. Mottaghi, M. Savva, A. R. Zamir.
   *On Evaluation of Embodied Navigation Agents.* arXiv:1807.06757, 2018.
2. J. Borenstein, Y. Koren. *Real-time Obstacle Avoidance for Fast Mobile
   Robots.* IEEE T-SMC, 19(5):1179–1187, 1989.
3. A.-C. Cheng et al. *NaVILA: Legged Robot Vision-Language-Action Model
   for Navigation.* arXiv:2412.04453, 2024.
4. R. C. Coulter. *Implementation of the Pure Pursuit Path Tracking
   Algorithm.* CMU-RI-TR-92-01, 1992.
5. D. Ha, J. Schmidhuber. *World Models.* NeurIPS, 2018.
6. D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering Diverse Domains
   through World Models (DreamerV3).* arXiv:2301.04104, 2023.
7. O. Khatib. *Real-Time Obstacle Avoidance for Manipulators and Mobile
   Robots.* IJRR, 5(1):90–98, 1986.
8. T. Liang et al. *BEVFusion: A Simple and Robust LiDAR-Camera Fusion
   Framework.* NeurIPS, 2023.
9. Y. Liu et al. *InternNav: A Foundation Navigation Agent for Embodied
   Tasks.* arXiv preprint, 2025.
10. Z. Liu et al. *OmniVLA: Omni-Modal Vision-Language-Action Model.*
    arXiv preprint, 2025.
11. J. Philion, S. Fidler. *Lift, Splat, Shoot: Encoding Images from
    Arbitrary Camera Rigs by Implicitly Unprojecting to 3D.* ECCV, 2020.
12. R. Simmons. *The Curvature-Velocity Method for Local Obstacle
    Avoidance.* ICRA, 1996.
13. F. Xia, A. R. Zamir, Z.-Y. He, A. Sax, J. Malik, S. Savarese.
    *Gibson Env: Real-World Perception for Embodied Agents.* CVPR, 2018.
