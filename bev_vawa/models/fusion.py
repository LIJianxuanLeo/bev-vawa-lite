"""Candidate fusion rule: combines VA policy score, WA risk/progress/uncertainty,
and the dead-end (reachability) estimate into one scalar ``Q`` per candidate.

    Q_k = α · softmax(s)_k + β · p_k − γ · σ(r_k) − δ · u_k − η · σ(d_k)

where ``s`` are the VA logits, ``p`` is predicted progress, ``r`` is the WA
risk logit, ``u`` is the ensemble uncertainty, and ``d`` is the dead-end
logit. At inference time the policy selects ``argmax_k Q_k``.
"""
from __future__ import annotations
import torch


def fuse_scores(
    va_logits: torch.Tensor,      # (B, K)
    wa_risk_logit: torch.Tensor,  # (B, K)
    wa_progress: torch.Tensor,    # (B, K)
    wa_unc: torch.Tensor,         # (B, K)
    deadend_logit: torch.Tensor,  # (B, K)
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    eta: float = 1.0,
) -> torch.Tensor:
    """Return per-candidate scalar Q (B, K). Higher is better."""
    risk = torch.sigmoid(wa_risk_logit)
    dead = torch.sigmoid(deadend_logit)
    policy = torch.softmax(va_logits, dim=-1)
    return alpha * policy + beta * wa_progress - gamma * risk - delta * wa_unc - eta * dead
