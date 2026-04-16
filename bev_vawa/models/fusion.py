"""Fusion rule: Q_k = alpha * s_k + beta * p_k - gamma * r_k - delta * u_k."""
from __future__ import annotations
import torch


def fuse_scores(
    va_logits: torch.Tensor,    # (B, K)
    wa_risk_logit: torch.Tensor, # (B, K)
    wa_progress: torch.Tensor,   # (B, K)
    wa_unc: torch.Tensor,        # (B, K)
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return per-candidate scalar Q (B, K). Higher is better."""
    risk = torch.sigmoid(wa_risk_logit)   # (B, K) in [0, 1]
    policy = torch.softmax(va_logits, dim=-1)
    return alpha * policy + beta * wa_progress - gamma * risk - delta * wa_unc
