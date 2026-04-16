"""End-to-end BEV-VAWA model: encoder + VA + WA + fusion."""
from __future__ import annotations
from typing import Optional
import torch
from torch import nn

from .bev_encoder import BEVEncoder
from .va_head import VAHead
from .wa_head import WAHead
from .fusion import fuse_scores


class BEVVAWA(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        bev = cfg["bev"]
        va = cfg["va"]
        wa = cfg["wa"]
        self.cfg = cfg
        self.encoder = BEVEncoder(
            depth_wh=tuple(cfg["env"]["depth_wh"]),
            grid_size=bev["grid_size"],
            latent_dim=bev["latent_dim"],
            cnn_channels=tuple(bev["cnn_channels"]),
            recurrent=bev["recurrent"],
        )
        self.va = VAHead(latent_dim=bev["latent_dim"], n_candidates=va["n_candidates"])
        self.wa = WAHead(
            latent_dim=bev["latent_dim"], n_candidates=va["n_candidates"],
            waypoint_embed_dim=wa["waypoint_embed_dim"],
            rollout_horizon=wa["rollout_horizon"],
            ensemble=wa["ensemble"],
        )
        # register anchors as buffer so they move with the model device
        from ..data.expert import candidate_anchors
        anchors = candidate_anchors(va["n_candidates"], va["waypoint_horizon_m"])
        self.register_buffer("anchors", torch.from_numpy(anchors), persistent=False)

    # ------------------------------------------------------------------
    def forward(self, depth: torch.Tensor, goal: torch.Tensor, use_wa: bool = True) -> dict:
        z = self.encoder(depth, goal)                                 # (B, latent)
        va_out = self.va(z)
        B = z.shape[0]
        # resolved candidate waypoints in robot frame: anchors (+ optional learned offset)
        anchors = self.anchors.to(z.dtype).unsqueeze(0).expand(B, -1, -1)
        if "offset" in va_out:
            waypoints = anchors + va_out["offset"]
        else:
            waypoints = anchors
        out = {"z": z, "va_logits": va_out["logits"], "waypoints": waypoints}
        if use_wa:
            wa_out = self.wa(z, waypoints)
            out.update({
                "wa_risk_logit": wa_out["risk_logit"],
                "wa_progress": wa_out["progress"],
                "wa_unc": wa_out["uncertainty"],
                "risk_ensemble": wa_out["risk_ensemble"],
            })
        return out

    def select_waypoint(self, out: dict, fusion_cfg: Optional[dict] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a forward() output, return (selected_waypoint (B, 2), selected_k (B,))."""
        cfg = fusion_cfg or self.cfg["fusion"]
        if "wa_risk_logit" in out:
            Q = fuse_scores(
                out["va_logits"], out["wa_risk_logit"], out["wa_progress"], out["wa_unc"],
                alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], delta=cfg["delta"],
            )
        else:
            Q = out["va_logits"]
        k_star = Q.argmax(dim=-1)                                     # (B,)
        wp = out["waypoints"].gather(1, k_star.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        return wp, k_star
