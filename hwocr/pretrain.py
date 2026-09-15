"""Warm start: train the CRNN as a single-letter classifier (CE loss).

The network is identical to the CTC recogniser - only the loss changes (the
per-frame logits are mean-pooled over time and classified with cross-entropy).
That means the trained weights can be loaded directly into the CRNN, giving the
CTC stage a CNN that already knows Hebrew letter shapes.

Usage::

    python -m hwocr.cli pretrain              # ~2 minutes on 6 CPU cores
    python -m hwocr.cli train --init-from models/letters_pretrained.pt
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import charset, dataset as ds, fonts, letters as letters_mod, model as model_mod
from .config import MODEL_DIR, Config

DEFAULT_PRETRAINED = MODEL_DIR / "letters_pretrained.pt"


def _collate(items: list[dict], width_reduction: int = 8) -> dict:
    """Pad the single-letter images of a batch to a common width."""
    max_w = max(item["image"].shape[-1] for item in items)
    max_w = max(width_reduction * 4, int(np.ceil(max_w / width_reduction) * width_reduction))
    height = items[0]["image"].shape[1]
    images = torch.full((len(items), 1, height, max_w), 1.0)  # 1.0 = white background
    for i, item in enumerate(items):
        width = item["image"].shape[-1]
        images[i, :, :, :width] = item["image"]
    return {
        "image": images,
        "target": torch.tensor([item["target"] for item in items], dtype=torch.long),
    }


def train_letters(
    cfg: Config | None = None,
    specs: Sequence[fonts.FontSpec] | None = None,
    steps: int = 800,
    batch_size: int = 32,
    lr: float = 2e-3,
    out_path: str | Path = DEFAULT_PRETRAINED,
    val_every: int = 200,
    target_accuracy: float = 0.99,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    """Train the CNN on isolated letters; returns metrics and the saved path."""
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed + seed)
    torch.set_num_threads(cfg.torch_threads)
    specs = list(specs) if specs else fonts.resolve_fonts(cfg)

    train_set = letters_mod.SingleLetterDataset(cfg, specs, seed=cfg.seed + seed,
                                                length=1_000_000, augment_prob=0.5)
    val_set = letters_mod.SingleLetterDataset(cfg, specs, seed=777, length=1000,
                                              augment_prob=0.0)
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=cfg.workers,
        collate_fn=_collate, worker_init_fn=ds.worker_init, drop_last=True,
    )

    model = model_mod.build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps,
                                                           eta_min=lr / 20.0)
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def validate() -> float:
        was_training = model.training
        model.eval()
        correct = total = 0
        for i in range(0, len(val_set), 64):
            batch = _collate([val_set[j] for j in range(i, min(i + 64, len(val_set)))])
            pooled = model(batch["image"]).mean(dim=0)
            correct += int((pooled.argmax(dim=1) == batch["target"]).sum())
            total += batch["target"].numel()
        if was_training:
            model.train()
        return correct / max(1, total)

    iterator = iter(loader)
    history: list[dict] = []
    start = time.time()
    for step in range(1, steps + 1):
        batch = next(iterator)
        pooled = model(batch["image"]).mean(dim=0)     # (B, C)
        loss = criterion(pooled, batch["target"])
        accuracy = float((pooled.argmax(dim=1) == batch["target"]).float().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if not quiet and (step % 25 == 0 or step == 1):
            print(f"[pretrain] step {step}/{steps} loss {float(loss.detach()):.3f} "
                  f"batch acc {accuracy:.3f} lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"({time.time() - start:.0f}s)", flush=True)

        if step % val_every == 0 or step == steps:
            accuracy = validate()
            history.append({"step": step, "val_accuracy": accuracy})
            if not quiet:
                print(f"[pretrain] step {step} letter accuracy {accuracy:.4f} "
                      f"({time.time() - start:.0f}s)", flush=True)
            if accuracy >= target_accuracy:
                break

    model.cpu()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "charset": charset.CHARS,
            "history": history,
            "letters": letters_mod.LETTER_ALPHABET,
        },
        out_path,
    )
    best = max((entry["val_accuracy"] for entry in history), default=0.0)
    if not quiet:
        print(f"[pretrain] letter accuracy {best:.4f}; saved to {out_path}", flush=True)
    return {"path": str(out_path), "accuracy": best, "history": history}

