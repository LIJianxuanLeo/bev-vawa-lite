"""Stage B: freeze the encoder, train the WA head on risk + progress."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ..data.dataset import NavShardDataset
from ..models import BEVVAWA
from ..utils import get_device, get_logger, set_seed
from .losses import wa_loss

log = get_logger(__name__)


def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_stage_b(cfg: dict, data_dir: str, in_ckpt: str, out_ckpt: str,
                  epochs: int | None = None, max_batches: int | None = None) -> dict:
    set_seed(cfg["seed"] + 1)
    device = get_device()
    ds = NavShardDataset(data_dir, depth_max=cfg["env"]["depth_max_m"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"])

    model = BEVVAWA(cfg).to(device)
    state = torch.load(in_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.va.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.wa.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    n_epochs = epochs or cfg["train"]["epochs_stage_b"]
    final = {}
    for ep in range(n_epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for bi, batch in enumerate(dl):
            batch = _to_device(batch, device)
            out = model(batch["depth"], batch["goal"], use_wa=True)
            loss = wa_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.wa.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            total += float(loss["loss"].detach()) * batch["depth"].shape[0]
            n_seen += batch["depth"].shape[0]
            if max_batches is not None and bi + 1 >= max_batches:
                break
        final = {"epoch": ep, "loss": total / max(n_seen, 1)}
        log.info(f"[stage-b] {final}")
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "stage": "b"}, out_ckpt)
    return final
