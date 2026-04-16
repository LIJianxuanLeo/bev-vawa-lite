"""WA branch: predicts per-anchor future risk, progress, and uncertainty.

Architecture:
  * anchor embedding: 2-vector (x, y) -> embed_dim via small MLP
  * rollout: LSTMCell rolling ``rollout_horizon`` latent steps conditioned on
    the anchor embedding; we take the final hidden state as the "future latent".
  * heads:
        - risk: scalar logit (BCE with cand_collision)
        - progress: scalar (MSE with cand_progress)
        - uncertainty: predicted via variance of a small ensemble of risk heads.
"""
from __future__ import annotations
import torch
from torch import nn


class WAHead(nn.Module):
    def __init__(self, latent_dim: int = 128, n_candidates: int = 5,
                 waypoint_embed_dim: int = 16, rollout_horizon: int = 3,
                 ensemble: int = 3):
        super().__init__()
        self.K = n_candidates
        self.H = rollout_horizon
        self.ensemble_n = ensemble

        self.anchor_embed = nn.Sequential(
            nn.Linear(2, waypoint_embed_dim), nn.ReLU(inplace=True),
            nn.Linear(waypoint_embed_dim, waypoint_embed_dim),
        )
        self.transition = nn.LSTMCell(waypoint_embed_dim, latent_dim)
        self.progress_head = nn.Linear(latent_dim, 1)
        self.risk_ensemble = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(ensemble)])

    def forward(self, z: torch.Tensor, anchors: torch.Tensor) -> dict:
        """z: (B, latent). anchors: (B, K, 2). Returns per-anchor (risk, progress, unc)."""
        B, K, _ = anchors.shape
        anchor_emb = self.anchor_embed(anchors)                      # (B, K, E)
        # expand latent to (B*K, latent) and roll out H steps
        h = z.unsqueeze(1).expand(B, K, z.shape[-1]).reshape(B * K, -1).contiguous()
        c = torch.zeros_like(h)
        inp = anchor_emb.reshape(B * K, -1).contiguous()
        for _ in range(self.H):
            h, c = self.transition(inp, (h, c))
        h = h.view(B, K, -1)                                         # (B, K, latent)
        # progress: single regression
        progress = self.progress_head(h).squeeze(-1)                 # (B, K)
        # risk: ensemble mean + variance
        risks = torch.stack([head(h).squeeze(-1) for head in self.risk_ensemble], dim=0)  # (M, B, K)
        risk_mean = risks.mean(dim=0)                                # (B, K) logits
        unc = risks.sigmoid().var(dim=0)                             # (B, K) in [0, 0.25]
        return {"risk_logit": risk_mean, "progress": progress, "uncertainty": unc,
                "risk_ensemble": risks}
