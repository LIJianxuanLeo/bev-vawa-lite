"""Stage C: joint fine-tune of last encoder block + both heads with combined loss."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ..data.dataset import NavShardDataset
from ..utils import get_device, get_logger, set_seed
from .losses import va_loss
from ._common import build_model, wa_loss_for_stage

log = get_logger(__name__)


def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_stage_c(cfg: dict, data_dir: str, in_ckpt: str, out_ckpt: str,
                  epochs: int | None = None, max_batches: int | None = None) -> dict:
    set_seed(cfg["seed"] + 2)
    device = get_device()
    ds = NavShardDataset(data_dir, depth_max=cfg["env"]["depth_max_m"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"])

    model = build_model(cfg).to(device)
    state = torch.load(in_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)

    # unfreeze only the last encoder block (bev_pool/fc_pool/input_proj) +
    # both heads. The earlier CNN blocks stay frozen to preserve low-level
    # geometry features learned in Stage A.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.bev_pool.parameters():
        p.requires_grad = True
    for p in model.encoder.fc_pool.parameters():
        p.requires_grad = True
    for p in model.encoder.input_proj.parameters():
        p.requires_grad = True
    for p in model.va.parameters():
        p.requires_grad = True
    for p in model.wa.parameters():
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"] * 0.3,
                            weight_decay=cfg["train"]["weight_decay"])
    n_epochs = epochs or cfg["train"]["epochs_stage_c"]
    final = {}
    for ep in range(n_epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for bi, batch in enumerate(dl):
            batch = _to_device(batch, device)
            out = model(batch["depth"], batch["goal"], use_wa=True)
            la = va_loss(out, batch)
            lb = wa_loss_for_stage(cfg, model, out, batch, stage="c")
            loss = la["loss"] + lb["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
            opt.step()
            total += float(loss.detach()) * batch["depth"].shape[0]
            n_seen += batch["depth"].shape[0]
            if max_batches is not None and bi + 1 >= max_batches:
                break
        final = {"epoch": ep, "loss": total / max(n_seen, 1)}
        log.info(f"[stage-c] {final}")
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "stage": "c"}, out_ckpt)
    return final
