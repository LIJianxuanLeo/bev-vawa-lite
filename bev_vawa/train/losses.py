"""Training losses for VA and WA heads."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def va_loss(out: dict, batch: dict) -> dict:
    """
    CE on best-k logits + Huber on the selected candidate's offset vs expert waypoint.
    """
    logits = out["va_logits"]                              # (B, K)
    best_k = batch["best_k"]                               # (B,)
    ce = F.cross_entropy(logits, best_k)

    wps = out["waypoints"]                                 # (B, K, 2)
    sel = wps.gather(1, best_k.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)  # (B, 2)
    tgt = batch["expert_wp"].to(sel.dtype)                 # (B, 2)
    huber = F.smooth_l1_loss(sel, tgt)
    total = ce + 0.5 * huber
    return {"loss": total, "ce": ce.detach(), "huber": huber.detach()}


def wa_loss(out: dict, batch: dict) -> dict:
    risk_logit = out["wa_risk_logit"]                      # (B, K)
    progress = out["wa_progress"]                          # (B, K)
    ensemble = out["risk_ensemble"]                        # (M, B, K)

    coll = batch["cand_collision"].to(risk_logit.dtype)    # (B, K) float
    prog = batch["cand_progress"].to(progress.dtype)       # (B, K)
    # every ensemble member learns the same target (bagging via random init only)
    l_risk = F.binary_cross_entropy_with_logits(risk_logit, coll)
    l_risk_ens = sum(
        F.binary_cross_entropy_with_logits(ensemble[i], coll) for i in range(ensemble.shape[0])
    ) / ensemble.shape[0]
    l_prog = F.mse_loss(progress, prog)
    total = l_risk + 0.5 * l_risk_ens + l_prog
    return {"loss": total, "risk": l_risk.detach(), "risk_ens": l_risk_ens.detach(), "prog": l_prog.detach()}
