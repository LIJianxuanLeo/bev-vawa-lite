"""World-Action (WA) head: candidate rollout with latent dynamics + dead-end.

For each of the K candidate waypoints the encoder latent ``z`` is rolled out
for ``H`` steps through an LSTMCell, conditioned on a learned anchor
embedding. Every intermediate hidden state is kept in
``z_hat: (B, K, H, latent)`` — this is the supervision target of the latent
dynamics loss ``L_dyn``.

From the final rollout hidden state we predict:
  * a **collision-risk** logit (plus an M-member ensemble for uncertainty),
  * a scalar **progress** regression target,
  * a **dead-end** logit (reachability), trained via BCE against offline
    pathfinder labels.
"""
from __future__ import annotations
import torch
from torch import nn


class WAHead(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        n_candidates: int = 5,
        waypoint_embed_dim: int = 16,
        rollout_horizon: int = 3,
        ensemble: int = 3,
    ) -> None:
        super().__init__()
        self.K = n_candidates
        self.H = rollout_horizon
        self.ensemble_n = ensemble
        self.latent_dim = latent_dim

        self.anchor_embed = nn.Sequential(
            nn.Linear(2, waypoint_embed_dim), nn.ReLU(inplace=True),
            nn.Linear(waypoint_embed_dim, waypoint_embed_dim),
        )
        self.transition = nn.LSTMCell(waypoint_embed_dim, latent_dim)
        self.progress_head = nn.Linear(latent_dim, 1)
        self.risk_ensemble = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(ensemble)])
        self.deadend_head = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor, anchors: torch.Tensor) -> dict:
        """z: (B, latent). anchors: (B, K, 2).

        Returns dict with
          * ``risk_logit``    (B, K)
          * ``progress``      (B, K)
          * ``uncertainty``   (B, K)
          * ``risk_ensemble`` (M, B, K)
          * ``z_hat``         (B, K, H, latent)
          * ``deadend_logit`` (B, K)
        """
        B, K, _ = anchors.shape
        H = self.H
        L = self.latent_dim

        anchor_emb = self.anchor_embed(anchors)                           # (B, K, E)
        h = z.unsqueeze(1).expand(B, K, L).reshape(B * K, L).contiguous()
        c = torch.zeros_like(h)
        inp = anchor_emb.reshape(B * K, -1).contiguous()

        z_hat_steps = []
        for _ in range(H):
            h, c = self.transition(inp, (h, c))
            z_hat_steps.append(h)
        z_hat = torch.stack(z_hat_steps, dim=0).view(H, B, K, L).permute(1, 2, 0, 3).contiguous()

        h_final = h.view(B, K, L)
        progress = self.progress_head(h_final).squeeze(-1)
        deadend_logit = self.deadend_head(h_final).squeeze(-1)
        risks = torch.stack(
            [head(h_final).squeeze(-1) for head in self.risk_ensemble], dim=0
        )
        risk_mean = risks.mean(dim=0)
        unc = risks.sigmoid().var(dim=0)

        return {
            "risk_logit": risk_mean,
            "progress": progress,
            "uncertainty": unc,
            "risk_ensemble": risks,
            "z_hat": z_hat,
            "deadend_logit": deadend_logit,
        }
