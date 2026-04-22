"""Training losses for the VA and WA heads.

* ``va_loss``   : cross-entropy on candidate logits + Huber on the selected
                  waypoint offset. Unchanged across stages.
* ``wa_loss``   : v1 risk + progress + ensemble BCE, plus the two v2 terms:

    - ``L_dyn``     : MSE between the WA rollout hidden state
                      ``z_hat[best_k]`` and a no-grad encoder target
                      ``z_gt_future`` on observed future frames. Aligns the
                      WA trajectory to actual encoder dynamics.
    - ``L_deadend`` : BCE on a per-candidate dead-end logit against offline
                      navmesh-pathfinder labels.

Both v2 terms are gated: when their weight is 0 or the required batch keys /
``z_gt_future`` are absent, they contribute 0 and the loss degrades cleanly
to the classical WA loss (useful on datasets whose shards don't carry
future frames, e.g. MuJoCo rollouts).
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def va_loss(out: dict, batch: dict) -> dict:
    """CE on best-k logits + Huber on the selected candidate's waypoint."""
    logits = out["va_logits"]                              # (B, K)
    best_k = batch["best_k"]                               # (B,)
    ce = F.cross_entropy(logits, best_k)

    wps = out["waypoints"]                                 # (B, K, 2)
    sel = wps.gather(1, best_k.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    tgt = batch["expert_wp"].to(sel.dtype)
    huber = F.smooth_l1_loss(sel, tgt)
    total = ce + 0.5 * huber
    return {"loss": total, "ce": ce.detach(), "huber": huber.detach()}


def wa_loss(
    out: dict,
    batch: dict,
    z_gt_future: Optional[torch.Tensor] = None,
    lambda_dyn: float = 0.5,
    lambda_deadend: float = 0.5,
    lambda_coll_head: float = 0.0,
) -> dict:
    """WA loss: risk + progress + ensemble + optional ``L_dyn`` + ``L_deadend``
    + optional learned collision head (``lambda_coll_head``).

    Parameters
    ----------
    out
        Forward output of ``BEVVAWA``. Must contain ``wa_risk_logit``,
        ``wa_progress``, ``risk_ensemble`` and (for the extended terms)
        ``z_hat`` ``(B, K, H, L)``, ``deadend_logit`` ``(B, K)``, and
        ``coll_logit_learned`` ``(B, K)``.
    batch
        Dataloader sample. Must contain ``cand_collision``, ``cand_progress``,
        ``best_k``, and — when ``lambda_deadend > 0`` — ``cand_deadend``.
    z_gt_future
        Ground-truth future encoder latents, shape ``(B, H, L)``.
    lambda_coll_head
        Weight on the learned collision BCE. Default 0 keeps the loss
        mathematically identical to the original WA loss; set to 0.3 on
        the Gibson / HM3D track to activate the head.
    """
    risk_logit = out["wa_risk_logit"]
    progress = out["wa_progress"]
    ensemble = out["risk_ensemble"]

    coll = batch["cand_collision"].to(risk_logit.dtype)
    prog = batch["cand_progress"].to(progress.dtype)
    l_risk = F.binary_cross_entropy_with_logits(risk_logit, coll)
    l_risk_ens = sum(
        F.binary_cross_entropy_with_logits(ensemble[i], coll) for i in range(ensemble.shape[0])
    ) / ensemble.shape[0]
    l_prog = F.mse_loss(progress, prog)

    # --- L_dyn: align rollout hidden of best_k with the true future latent --
    if lambda_dyn > 0 and z_gt_future is not None and "z_hat" in out:
        z_hat = out["z_hat"]                                    # (B, K, H, L)
        best_k = batch["best_k"].view(-1, 1, 1, 1)
        best_k = best_k.expand(-1, 1, z_hat.shape[2], z_hat.shape[3])
        z_hat_best = z_hat.gather(1, best_k).squeeze(1)         # (B, H, L)
        l_dyn = F.mse_loss(z_hat_best, z_gt_future.to(z_hat_best.dtype))
    else:
        l_dyn = risk_logit.new_zeros(())

    # --- L_deadend: per-candidate BCE ---------------------------------------
    if lambda_deadend > 0 and "cand_deadend" in batch and "deadend_logit" in out:
        dead_target = batch["cand_deadend"].to(risk_logit.dtype)
        l_deadend = F.binary_cross_entropy_with_logits(out["deadend_logit"], dead_target)
    else:
        l_deadend = risk_logit.new_zeros(())

    # --- L_coll_head: learned collision-probability BCE ---------------------
    if lambda_coll_head > 0 and "coll_logit_learned" in out:
        l_coll_head = F.binary_cross_entropy_with_logits(
            out["coll_logit_learned"], coll
        )
    else:
        l_coll_head = risk_logit.new_zeros(())

    total = (
        l_risk + 0.5 * l_risk_ens + l_prog
        + lambda_dyn * l_dyn
        + lambda_deadend * l_deadend
        + lambda_coll_head * l_coll_head
    )
    return {
        "loss": total,
        "risk": l_risk.detach(),
        "risk_ens": l_risk_ens.detach(),
        "prog": l_prog.detach(),
        "dyn": l_dyn.detach(),
        "deadend": l_deadend.detach(),
        "coll_head": l_coll_head.detach(),
    }
