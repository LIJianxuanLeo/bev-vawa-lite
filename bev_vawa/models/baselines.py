"""Baseline models required by the paper's §9.1.

Each baseline shares the VA-style output (K logits + selected waypoint) so the
same closed-loop evaluator can execute them. The differences are architectural:

* ``FPV_BC``   : first-person depth -> flat MLP -> K logits. No BEV, no WA.
* ``BEV_BC``   : BEV encoder -> direct (v, omega) regression (no candidate list).
* ``BEV_VA``   : BEV encoder + VA head, no WA branch.
* ``BEVVAWA``  : full model (see models/full_model.py).
* A*-upper-bound is not learned — see ``eval/policies.py::make_astar_policy``.
"""
from __future__ import annotations
import torch
from torch import nn

from .bev_encoder import BEVEncoder
from .va_head import VAHead
from ..data.expert import candidate_anchors


class FPV_BC(nn.Module):
    """Flat CNN over first-person depth, concatenated with goal -> K logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        H, W = cfg["env"]["depth_wh"]
        K = cfg["va"]["n_candidates"]
        self.K = K
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 16 + 2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, K),
        )
        anchors = candidate_anchors(K, cfg["va"]["waypoint_horizon_m"])
        self.register_buffer("anchors", torch.from_numpy(anchors), persistent=False)

    def forward(self, depth, goal, use_wa: bool = False):
        f = self.cnn(depth).flatten(1)
        logits = self.fc(torch.cat([f, goal], dim=-1))
        B = f.shape[0]
        waypoints = self.anchors.to(f.dtype).unsqueeze(0).expand(B, -1, -1)
        return {"z": f, "va_logits": logits, "waypoints": waypoints}

    def select_waypoint(self, out, fusion_cfg=None):
        k = out["va_logits"].argmax(dim=-1)
        wp = out["waypoints"].gather(1, k.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        return wp, k


class BEV_VA(nn.Module):
    """BEV encoder + VA only (no WA). Used as a direct ablation."""

    def __init__(self, cfg: dict):
        super().__init__()
        bev = cfg["bev"]
        va = cfg["va"]
        self.encoder = BEVEncoder(
            depth_wh=tuple(cfg["env"]["depth_wh"]),
            grid_size=bev["grid_size"],
            latent_dim=bev["latent_dim"],
            cnn_channels=tuple(bev["cnn_channels"]),
            recurrent=bev["recurrent"],
        )
        self.va = VAHead(latent_dim=bev["latent_dim"], n_candidates=va["n_candidates"])
        anchors = candidate_anchors(va["n_candidates"], va["waypoint_horizon_m"])
        self.register_buffer("anchors", torch.from_numpy(anchors), persistent=False)

    def forward(self, depth, goal, use_wa: bool = False):
        z = self.encoder(depth, goal)
        va_out = self.va(z)
        B = z.shape[0]
        anchors = self.anchors.to(z.dtype).unsqueeze(0).expand(B, -1, -1)
        waypoints = anchors + va_out.get("offset", torch.zeros_like(anchors))
        return {"z": z, "va_logits": va_out["logits"], "waypoints": waypoints}

    def select_waypoint(self, out, fusion_cfg=None):
        k = out["va_logits"].argmax(dim=-1)
        wp = out["waypoints"].gather(1, k.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        return wp, k


class BEV_BC(nn.Module):
    """BEV encoder that regresses (waypoint_x, waypoint_y) directly — no candidates."""

    def __init__(self, cfg: dict):
        super().__init__()
        bev = cfg["bev"]
        self.encoder = BEVEncoder(
            depth_wh=tuple(cfg["env"]["depth_wh"]),
            grid_size=bev["grid_size"],
            latent_dim=bev["latent_dim"],
            cnn_channels=tuple(bev["cnn_channels"]),
            recurrent=bev["recurrent"],
        )
        self.head = nn.Sequential(
            nn.Linear(bev["latent_dim"], 128), nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        # dummy K=1 for compatibility with closed-loop eval
        self.K = 1

    def forward(self, depth, goal, use_wa: bool = False):
        z = self.encoder(depth, goal)
        wp = self.head(z).view(-1, 1, 2)
        logits = torch.zeros(wp.shape[0], 1, device=wp.device, dtype=wp.dtype)
        return {"z": z, "va_logits": logits, "waypoints": wp}

    def select_waypoint(self, out, fusion_cfg=None):
        k = torch.zeros(out["waypoints"].shape[0], dtype=torch.long, device=out["waypoints"].device)
        wp = out["waypoints"][:, 0]
        return wp, k
