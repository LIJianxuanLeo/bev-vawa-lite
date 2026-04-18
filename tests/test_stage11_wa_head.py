"""Stage 11 gate: WA head (latent dynamics + dead-end head).

Covers:
  * ``WAHead`` forward shape (per-step ``z_hat``, ``deadend_logit``).
  * ``wa_loss`` terms — ``L_dyn`` on the expert candidate, ``L_deadend`` BCE.
  * ``fuse_scores`` — the ``-η·d`` term changes ranking when dead-end
    probability dominates.
  * End-to-end backward through ``BEVVAWA`` with ``wa_loss``.

Runs on every host (pure PyTorch).
"""
from __future__ import annotations
import torch

from bev_vawa.models import BEVVAWA, WAHead, fuse_scores
from bev_vawa.train.losses import wa_loss
from bev_vawa.utils import load_config


# ---------------------------------------------------------------- WAHead
def test_wa_head_forward_shapes():
    head = WAHead(latent_dim=64, n_candidates=5, rollout_horizon=3, ensemble=3)
    z = torch.randn(4, 64)
    anchors = torch.randn(4, 5, 2)
    out = head(z, anchors)
    assert out["risk_logit"].shape == (4, 5)
    assert out["progress"].shape == (4, 5)
    assert out["uncertainty"].shape == (4, 5)
    assert out["risk_ensemble"].shape == (3, 4, 5)
    assert out["z_hat"].shape == (4, 5, 3, 64)
    assert out["deadend_logit"].shape == (4, 5)


# ---------------------------------------------------------------- wa_loss
def _mk_batch(B: int = 4, K: int = 5):
    return {
        "best_k": torch.randint(0, K, (B,)),
        "cand_collision": torch.randint(0, 2, (B, K)).float(),
        "cand_progress": torch.rand(B, K),
        "cand_deadend": torch.randint(0, 2, (B, K)).float(),
    }


def _mk_out(B: int = 4, K: int = 5, H: int = 3, L: int = 32):
    return {
        "wa_risk_logit": torch.randn(B, K, requires_grad=True),
        "wa_progress": torch.randn(B, K, requires_grad=True),
        "risk_ensemble": torch.randn(3, B, K, requires_grad=True),
        "z_hat": torch.randn(B, K, H, L, requires_grad=True),
        "deadend_logit": torch.randn(B, K, requires_grad=True),
    }


def test_wa_loss_has_all_terms():
    B, K, H, L = 4, 5, 3, 32
    out = _mk_out(B, K, H, L)
    batch = _mk_batch(B, K)
    z_gt = torch.randn(B, H, L)
    res = wa_loss(out, batch, z_gt_future=z_gt, lambda_dyn=0.5, lambda_deadend=0.5)
    for k in ("loss", "risk", "risk_ens", "prog", "dyn", "deadend"):
        assert k in res, f"wa_loss missing key {k!r}"
    assert torch.isfinite(res["loss"])
    assert res["dyn"].item() > 0
    assert res["deadend"].item() > 0


def test_wa_loss_dyn_drops_on_perfect_target():
    """If ``z_hat[best_k]`` equals ``z_gt_future`` exactly, L_dyn should be ~0."""
    B, K, H, L = 2, 3, 2, 8
    batch = _mk_batch(B, K)
    best_k = batch["best_k"]

    z_gt = torch.randn(B, H, L)
    z_hat = torch.zeros(B, K, H, L)
    for b in range(B):
        z_hat[b, best_k[b]] = z_gt[b]

    out = _mk_out(B, K, H, L)
    out["z_hat"] = z_hat.requires_grad_(True)
    res = wa_loss(out, batch, z_gt_future=z_gt, lambda_dyn=1.0, lambda_deadend=0.0)
    assert res["dyn"].item() < 1e-6


def test_wa_loss_lambda_zero_skips_terms():
    B, K, H, L = 2, 3, 2, 8
    out = _mk_out(B, K, H, L)
    batch = _mk_batch(B, K)
    # no future latent provided + lambda_dyn=0 → dyn term must be 0
    res = wa_loss(out, batch, z_gt_future=None, lambda_dyn=0.0, lambda_deadend=0.0)
    assert res["dyn"].item() == 0.0
    assert res["deadend"].item() == 0.0


def test_wa_loss_converges_on_toy():
    """Short toy fit: L_dyn + L_deadend + L_risk should decrease."""
    torch.manual_seed(0)
    B, K, H, L = 8, 4, 2, 16
    head = WAHead(latent_dim=L, n_candidates=K, rollout_horizon=H, ensemble=2)
    z = torch.randn(B, L)
    anchors = torch.randn(B, K, 2)
    batch = _mk_batch(B, K)
    z_gt = torch.randn(B, H, L)

    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    losses = []
    for _ in range(80):
        wa = head(z, anchors)
        out = {
            "wa_risk_logit": wa["risk_logit"],
            "wa_progress": wa["progress"],
            "risk_ensemble": wa["risk_ensemble"],
            "z_hat": wa["z_hat"],
            "deadend_logit": wa["deadend_logit"],
        }
        res = wa_loss(out, batch, z_gt_future=z_gt,
                     lambda_dyn=0.5, lambda_deadend=0.5)
        opt.zero_grad()
        res["loss"].backward()
        opt.step()
        losses.append(float(res["loss"].detach()))

    assert losses[-1] < losses[0] * 0.75, f"loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------- fusion
def test_fuse_scores_deadend_flips_ranking():
    """With η large and one candidate confidently dead-end, that candidate
    must rank last even if its VA / progress were winning."""
    B, K = 1, 3
    va_logits = torch.tensor([[5.0, 0.0, 0.0]])
    wa_risk = torch.zeros(B, K)
    wa_prog = torch.zeros(B, K)
    wa_unc = torch.zeros(B, K)
    deadend_logit = torch.tensor([[5.0, -5.0, -5.0]])
    Q = fuse_scores(va_logits, wa_risk, wa_prog, wa_unc, deadend_logit,
                    alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, eta=5.0)
    assert Q.argmax(dim=-1).item() != 0
    assert Q.argmin(dim=-1).item() == 0


def test_fuse_scores_eta_zero_removes_deadend_term():
    """With η=0, the dead-end column drops out entirely."""
    B, K = 2, 5
    torch.manual_seed(1)
    va = torch.randn(B, K)
    risk = torch.randn(B, K)
    prog = torch.randn(B, K)
    unc = torch.rand(B, K)
    dead_a = torch.randn(B, K)
    dead_b = torch.randn(B, K)
    Qa = fuse_scores(va, risk, prog, unc, dead_a,
                     alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, eta=0.0)
    Qb = fuse_scores(va, risk, prog, unc, dead_b,
                     alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, eta=0.0)
    assert torch.allclose(Qa, Qb, atol=1e-6)


# ---------------------------------------------------------------- BEVVAWA e2e
def test_bev_vawa_end_to_end_backward():
    """Full forward + wa_loss backward exercises every code path:
    geometry-lift encoder, WA head (with z_hat, deadend_logit), encode_future,
    L_dyn, L_deadend."""
    cfg = load_config("configs/habitat/gibson.yaml")
    model = BEVVAWA(cfg)

    B = 2
    H = cfg["wa"]["rollout_horizon"]
    Hi, Wi = cfg["env"]["depth_wh"]
    depth = torch.rand(B, 1, Hi, Wi) * 2.0
    goal = torch.zeros(B, 2)
    future_depth = torch.rand(B, H, 1, Hi, Wi) * 2.0
    future_goal = torch.zeros(B, H, 2)

    out = model(depth, goal, use_wa=True,
                future_depth=future_depth, future_goal=future_goal)
    for k in ("z", "va_logits", "waypoints", "wa_risk_logit", "wa_progress",
              "wa_unc", "risk_ensemble", "z_hat", "deadend_logit", "z_gt_future"):
        assert k in out, f"BEVVAWA forward missing key {k!r}"

    K = out["va_logits"].shape[1]
    batch = {
        "best_k": torch.zeros(B, dtype=torch.long),
        "cand_collision": torch.zeros(B, K),
        "cand_progress": torch.zeros(B, K),
        "cand_deadend": torch.zeros(B, K),
    }
    res = wa_loss(out, batch, z_gt_future=out["z_gt_future"],
                  lambda_dyn=0.5, lambda_deadend=0.5)
    res["loss"].backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads in {bad[:5]}"


def test_bev_vawa_select_waypoint_uses_deadend_term():
    """When deadend_logit is present, select_waypoint must route through
    fuse_scores (so the -η·d term participates in ranking)."""
    cfg = load_config("configs/habitat/gibson.yaml")
    model = BEVVAWA(cfg).eval()
    B = 1
    Hi, Wi = cfg["env"]["depth_wh"]
    with torch.no_grad():
        out = model(torch.rand(B, 1, Hi, Wi), torch.zeros(B, 2), use_wa=True)
        K = out["va_logits"].shape[1]
        out["va_logits"] = torch.zeros(B, K)
        out["wa_progress"] = torch.zeros(B, K)
        out["wa_risk_logit"] = torch.zeros(B, K)
        out["wa_unc"] = torch.zeros(B, K)
        dead = torch.full((B, K), -5.0)
        dead[:, 0] = 5.0
        out["deadend_logit"] = dead
        out["waypoints"] = torch.zeros(B, K, 2)
        wp, k_star = model.select_waypoint(out, cfg["fusion"])
        assert wp.shape == (B, 2)
        assert k_star.item() != 0
