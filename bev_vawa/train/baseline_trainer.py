"""Generic trainer that works with FPV_BC, BEV_VA, BEV_BC.
Uses va_loss (best-k CE + Huber on selected waypoint) when K>1, and plain MSE
on waypoint-regression for BEV_BC."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import NavShardDataset
from ..utils import get_device, get_logger, set_seed

log = get_logger(__name__)


def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_baseline(model_cls, cfg: dict, data_dir: str, out_ckpt: str,
                   epochs: int | None = None, max_batches: int | None = None) -> dict:
    set_seed(cfg["seed"] + 77)
    device = get_device()
    ds = NavShardDataset(data_dir, depth_max=cfg["env"]["depth_max_m"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"])

    model = model_cls(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    n_epochs = epochs or cfg["train"]["epochs_stage_a"]
    final = {}
    for ep in range(n_epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for bi, batch in enumerate(dl):
            batch = _to_device(batch, device)
            out = model(batch["depth"], batch["goal"])
            if out["waypoints"].shape[1] == 1:
                # regression baseline
                loss = F.smooth_l1_loss(out["waypoints"][:, 0], batch["expert_wp"].to(out["waypoints"].dtype))
            else:
                ce = F.cross_entropy(out["va_logits"], batch["best_k"])
                sel = out["waypoints"].gather(
                    1, batch["best_k"].view(-1, 1, 1).expand(-1, 1, 2)
                ).squeeze(1)
                h = F.smooth_l1_loss(sel, batch["expert_wp"].to(sel.dtype))
                loss = ce + 0.5 * h
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            total += float(loss.detach()) * batch["depth"].shape[0]
            n_seen += batch["depth"].shape[0]
            if max_batches is not None and bi + 1 >= max_batches:
                break
        final = {"epoch": ep, "loss": total / max(n_seen, 1)}
        log.info(f"[baseline {model_cls.__name__}] {final}")
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "baseline": model_cls.__name__}, out_ckpt)
    return final
