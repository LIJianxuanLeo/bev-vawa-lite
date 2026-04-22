"""Candidate fusion rule: combines VA policy score, WA risk/progress/uncertainty,
dead-end (reachability), and — optionally — a learned short-horizon
collision probability into one scalar ``Q`` per candidate.

    Q_k = α · softmax(s)_k + β · p_k − γ · σ(r_k) − δ · u_k − η · σ(d_k)
          − μ · σ(c_k)

where ``s`` are the VA logits, ``p`` is predicted progress, ``r`` is the WA
risk logit, ``u`` is the ensemble uncertainty, ``d`` is the dead-end logit,
and ``c`` is the learned collision logit (``coll_logit_learned``). When
the collision head is not trained / not available, pass ``mu = 0`` and
the term drops out. At inference the policy selects ``argmax_k Q_k``.
"""
from __future__ import annotations
from typing import Optional

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
    coll_logit_learned: Optional[torch.Tensor] = None,  # (B, K) or None
    mu: float = 0.0,
) -> torch.Tensor:
    """Return per-candidate scalar Q (B, K). Higher is better.

    Setting ``mu=0`` or leaving ``coll_logit_learned=None`` disables the
    learned-collision term and recovers the original five-term fusion
    exactly (backward compatible with v2 checkpoints).
    """
    risk = torch.sigmoid(wa_risk_logit)
    dead = torch.sigmoid(deadend_logit)
    policy = torch.softmax(va_logits, dim=-1)
    Q = alpha * policy + beta * wa_progress - gamma * risk - delta * wa_unc - eta * dead
    if mu != 0.0 and coll_logit_learned is not None:
        Q = Q - mu * torch.sigmoid(coll_logit_learned)
    return Q
