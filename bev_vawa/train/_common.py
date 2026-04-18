"""Shared helpers for the stage-A / stage-B / stage-C trainers.

Centralises two things the trainers would otherwise duplicate:

* ``build_model(cfg)`` — instantiates the BEV-VAWA model.
* ``wa_loss_for_stage(cfg, model, out, batch, stage)`` — runs the WA loss
  with stage-appropriate dynamics / dead-end weights. In stage C the latent
  dynamics and dead-end terms are damped by 0.4 (per the design doc §5.6)
  so they don't overwhelm the gentler joint fine-tune.
"""
from __future__ import annotations
from typing import Optional

import torch

from ..models import BEVVAWA
from .losses import wa_loss


def build_model(cfg: dict) -> torch.nn.Module:
    return BEVVAWA(cfg)


def wa_loss_for_stage(cfg: dict, model: torch.nn.Module, out: dict, batch: dict,
                      stage: str = "b") -> dict:
    """Compute the WA loss for a given training stage.

    The configurable ``wa.enable_dyn`` / ``wa.enable_deadend`` flags control
    whether those terms participate at all; ``wa.lambda_dyn`` /
    ``wa.lambda_deadend`` set their weight. Stage C multiplies both weights
    by 0.4 to ease the joint fine-tune.
    """
    wa_cfg = cfg.get("wa", {})
    lam_dyn = float(wa_cfg.get("lambda_dyn", 0.5)) if wa_cfg.get("enable_dyn", True) else 0.0
    lam_dead = float(wa_cfg.get("lambda_deadend", 0.5)) if wa_cfg.get("enable_deadend", True) else 0.0
    if stage == "c":
        lam_dyn *= 0.4
        lam_dead *= 0.4

    z_gt_future: Optional[torch.Tensor] = None
    if lam_dyn > 0 and "future_depth" in batch and "future_goal" in batch:
        with torch.no_grad():
            z_gt_future = model.encode_future(batch["future_depth"], batch["future_goal"])

    return wa_loss(
        out, batch, z_gt_future=z_gt_future,
        lambda_dyn=lam_dyn, lambda_deadend=lam_dead,
    )
