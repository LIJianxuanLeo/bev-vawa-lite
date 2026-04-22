"""End-to-end BEV-VAWA model: geometric BEV encoder + VA head + WA head + fusion.

Public API:
  * ``forward(depth, goal, use_wa=True, future_depth=None, future_goal=None) -> dict``
  * ``encode_future(future_depth, future_goal) -> (B, H, latent)`` — no-grad,
    used by the trainer to produce the dynamics target ``z_gt_future``.
  * ``select_waypoint(out, fusion_cfg) -> (wp, k_star)`` — fuse candidate
    scores and pick the best k at inference time.
"""
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
        env = cfg["env"]
        self.cfg = cfg

        channels_enabled = tuple(bev.get("channels_enabled", (1, 1, 1)))
        bev_range = tuple(bev.get("range", (0.0, 3.0, -1.5, 1.5)))
        goal_sigma = float(bev.get("goal_sector_sigma_rad", 0.35))

        self.encoder = BEVEncoder(
            depth_wh=tuple(env["depth_wh"]),
            grid_size=bev["grid_size"],
            latent_dim=bev["latent_dim"],
            cnn_channels=tuple(bev["cnn_channels"]),
            recurrent=bev.get("recurrent", "lstm"),
            fov_deg=float(env["depth_fov_deg"]),
            depth_max_m=float(env["depth_max_m"]),
            bev_range=bev_range,
            channels_enabled=channels_enabled,
            goal_sector_sigma_rad=goal_sigma,
            use_semantic=bool(bev.get("use_semantic", False)),
            semantic_classes=int(bev.get("semantic_classes", 16)),
            semantic_feat_dim=int(bev.get("semantic_feat_dim", 64)),
            use_geometric_lift=bool(bev.get("use_geometric_lift", True)),
        )
        self.va = VAHead(latent_dim=bev["latent_dim"], n_candidates=va["n_candidates"])
        self.wa = WAHead(
            latent_dim=bev["latent_dim"],
            n_candidates=va["n_candidates"],
            waypoint_embed_dim=wa.get("waypoint_embed_dim", 16),
            rollout_horizon=wa["rollout_horizon"],
            ensemble=wa["ensemble"],
        )

        from ..data.expert import candidate_anchors
        anchors = candidate_anchors(va["n_candidates"], va["waypoint_horizon_m"])
        self.register_buffer("anchors", torch.from_numpy(anchors), persistent=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        depth: torch.Tensor,
        goal: torch.Tensor,
        use_wa: bool = True,
        future_depth: Optional[torch.Tensor] = None,
        future_goal: Optional[torch.Tensor] = None,
        semantic: Optional[torch.Tensor] = None,
        future_semantic: Optional[torch.Tensor] = None,
    ) -> dict:
        z = self.encoder(depth, goal, semantic=semantic)
        va_out = self.va(z)
        B = z.shape[0]
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
                "z_hat": wa_out["z_hat"],
                "deadend_logit": wa_out["deadend_logit"],
                "coll_logit_learned": wa_out["coll_logit_learned"],
            })
        if future_depth is not None and future_goal is not None:
            out["z_gt_future"] = self.encode_future(
                future_depth, future_goal, future_semantic=future_semantic
            )
        return out

    @torch.no_grad()
    def encode_future(self, future_depth: torch.Tensor,
                      future_goal: torch.Tensor,
                      future_semantic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the encoder (no-grad) on each future step to produce the
        dynamics target for the WA ``L_dyn`` loss.

        Parameters
        ----------
        future_depth : (B, H, 1, Hi, Wi)
        future_goal  : (B, H, 2)
        future_semantic : (B, H, Hi, Wi) int  *or*  (B, H, C, Hi, Wi) one-hot
                         optional; passed only when the encoder was built
                         with ``use_semantic=True``.

        Returns
        -------
        z_future : (B, H, latent)
        """
        assert future_depth.dim() == 5, f"expected (B,H,1,Hi,Wi), got {tuple(future_depth.shape)}"
        B, H, _, Hi, Wi = future_depth.shape
        flat_d = future_depth.reshape(B * H, 1, Hi, Wi)
        flat_g = future_goal.reshape(B * H, 2)
        flat_s = None
        if future_semantic is not None:
            if future_semantic.dim() == 4:       # (B, H, Hi, Wi) int map
                flat_s = future_semantic.reshape(B * H, Hi, Wi)
            elif future_semantic.dim() == 5:     # (B, H, C, Hi, Wi) one-hot
                C = future_semantic.shape[2]
                flat_s = future_semantic.reshape(B * H, C, Hi, Wi)
        z = self.encoder.encode_single(flat_d, flat_g, semantic=flat_s)  # (B*H, latent)
        return z.view(B, H, -1)

    # ------------------------------------------------------------------
    def select_waypoint(self, out: dict, fusion_cfg: Optional[dict] = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = fusion_cfg or self.cfg["fusion"]
        eta = float(cfg.get("eta", 1.0))
        mu = float(cfg.get("mu", 0.0))
        coll_logit = out.get("coll_logit_learned")
        if "wa_risk_logit" in out and "deadend_logit" in out:
            Q = fuse_scores(
                out["va_logits"], out["wa_risk_logit"], out["wa_progress"],
                out["wa_unc"], out["deadend_logit"],
                alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
                delta=cfg["delta"], eta=eta,
                coll_logit_learned=coll_logit, mu=mu,
            )
        elif "wa_risk_logit" in out:
            # WA was run without the dead-end branch — fall back to zero dead-end
            zero_dead = torch.zeros_like(out["wa_risk_logit"])
            Q = fuse_scores(
                out["va_logits"], out["wa_risk_logit"], out["wa_progress"],
                out["wa_unc"], zero_dead,
                alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
                delta=cfg["delta"], eta=0.0,
                coll_logit_learned=coll_logit, mu=mu,
            )
        else:
            Q = out["va_logits"]
        k_star = Q.argmax(dim=-1)
        wp = out["waypoints"].gather(1, k_star.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        return wp, k_star
