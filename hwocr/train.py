"""Training loop.

Two stages with different augmentation strength:

1. ``steps_light`` - clean renderings with mild jitter.  This is where the
   model learns the letter shapes, so it converges very quickly.
2. ``steps_heavy`` - photo-like augmentations mixed in (``cfg.p_heavy_aug``),
   which buys robustness on real images.

The validation set is generated from a fixed seed, so the reported numbers are
directly comparable between runs.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import charset, dataset as ds, decode, fonts, metrics, model as model_mod
from .config import DEFAULT_CHECKPOINT, RUNS_DIR, Config


def build_datasets(
    cfg: Config,
    specs: Sequence[fonts.FontSpec] | None = None,
    words: Sequence[str] | None = None,
) -> tuple[ds.SyntheticLineDataset, ds.SyntheticLineDataset]:
    specs = list(specs) if specs else fonts.resolve_fonts(cfg)
    train_set = ds.SyntheticLineDataset(cfg, specs, seed=cfg.seed, length=1_000_000,
                                        heavy_prob=0.0, words=words)
    val_set = ds.SyntheticLineDataset(cfg, specs, seed=ds.VAL_SEED, length=cfg.val_samples,
                                      heavy_prob=cfg.p_heavy_aug, words=words)
    return train_set, val_set


@torch.no_grad()
def evaluate(model: nn.Module, cfg: Config, loader, device: torch.device) -> metrics.Report:
    """CER/WER over a loader (labels are compared in logical/Hebrew order)."""
    was_training = model.training
    model.eval()
    report = metrics.Report()
    for batch in loader:
        images = batch["image"].to(device)
        log_probs = model(images)
        lengths = ds.ctc_input_lengths(batch["valid_width"], cfg.width_reduction,
                                       log_probs.shape[0]).tolist()
        decoded = decode.greedy_decode(log_probs.cpu(), lengths)
        for item, reference in zip(decoded, batch["logical_text"]):
            report.add(reference, item.logical, item.confidence)
    if was_training:
        model.train()
    return report


def _lr_at(step: int, cfg: Config) -> float:
    total = max(1, cfg.steps_short + cfg.steps_medium + cfg.steps_light + cfg.steps_heavy)
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def train(
    cfg: Config | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    extra_fonts: Sequence[str] = (),
    extra_dirs: Sequence[str] = (),
    include_system: bool = False,
    words: Sequence[str] | None = None,
    log_path: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train a recogniser; returns a dict with the training history."""
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    device = torch.device("cpu")

    specs = fonts.resolve_fonts(cfg, extra_fonts, extra_dirs, include_system)
    fonts_info = fonts.describe(specs, cfg.font_weights)
    if not quiet:
        print("[train] fonts:\n" + fonts_info, flush=True)

    train_set, val_set = build_datasets(cfg, specs, words)
    val_loader = ds.make_loader(val_set, cfg, shuffle=False, workers=0)

    model = model_mod.build_model(cfg).to(device)
    if not quiet:
        print(f"[train] parameters: {model_mod.count_parameters(model)/1e6:.2f}M, "
              f"classes: {charset.num_classes()}, input: {cfg.img_height}x{cfg.max_width}",
              flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=charset.BLANK_INDEX, zero_infinity=True)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"cer": 1.0, "wer": 1.0, "step": 0, "report": "-"}
    log_file = None
    writer = None
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["step", "stage", "loss", "lr", "images_per_s", "val_cer", "val_wer",
                         "val_line_acc", "elapsed_s"])

    step = 0
    start = time.time()
    seen_images = 0
    total_steps = cfg.steps_short + cfg.steps_medium + cfg.steps_light + cfg.steps_heavy
    # (name, steps, heavy_prob, augment_prob, max_words, max_word_len)
    stages = [
        ("short", cfg.steps_short, 0.0, 0.0,
         cfg.steps_short_max_words, cfg.steps_short_max_word_len),
        ("medium", cfg.steps_medium, 0.0, 0.0,
         cfg.steps_medium_max_words, cfg.steps_medium_max_word_len),
        ("light", cfg.steps_light, 0.0, 1.0, cfg.max_words, cfg.max_word_len),
        ("heavy", cfg.steps_heavy, cfg.p_heavy_aug, 1.0, cfg.max_words, cfg.max_word_len),
    ]

    try:
        for stage_name, stage_steps, heavy_prob, augment_prob, max_words, max_word_len in stages:
            if stage_steps <= 0:
                continue
            # Workers are forked when the iterator is created, so a fresh
            # loader is needed for the stage settings to take effect.
            train_set.heavy_prob = heavy_prob
            train_set.augment_prob = augment_prob
            train_set.max_words = max_words
            train_set.max_word_len = max_word_len
            loader = ds.make_loader(train_set, cfg, shuffle=True)
            iterator = iter(loader)
            if not quiet:
                print(f"[train] stage '{stage_name}': {stage_steps} steps, "
                      f"p(aug)={augment_prob}, p(heavy)={heavy_prob}, "
                      f"max words={max_words}, max word length={max_word_len}", flush=True)

            for _ in range(stage_steps):
                batch = next(iterator)
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                label_lengths = batch["label_length"].to(device)
                lr = _lr_at(step, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                log_probs = model(images)
                input_lengths = ds.ctc_input_lengths(
                    batch["valid_width"], cfg.width_reduction, log_probs.shape[0]
                ).to(device)
                # CTC needs at least (2 * label_length + 1) frames per sample.
                feasible = (2 * label_lengths + 1) <= input_lengths
                loss_value = float("nan")
                if bool(feasible.any()):
                    if bool(feasible.all()):
                        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    else:
                        loss = criterion(log_probs[:, feasible], labels[feasible],
                                         input_lengths[feasible], label_lengths[feasible])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    loss_value = float(loss.detach())

                seen_images += images.shape[0]
                elapsed = time.time() - start
                step += 1

                if not quiet and (step % cfg.log_every == 0 or step == 1):
                    print(f"[train] step {step}/{total_steps} loss={loss_value:.4f} "
                          f"lr={lr:.2e} {seen_images/elapsed:.0f} img/s", flush=True)

                if step % cfg.val_every == 0 or step == total_steps:
                    report = evaluate(model, cfg, val_loader, device)
                    record = {
                        "step": step, "stage": stage_name, "loss": loss_value, "lr": lr,
                        "images_per_s": seen_images / max(1e-6, elapsed),
                        "val_cer": report.cer, "val_wer": report.wer,
                        "val_line_acc": report.line_accuracy, "elapsed_s": elapsed,
                    }
                    history.append(record)
                    if writer:
                        writer.writerow([record[key] for key in
                                         ("step", "stage", "loss", "lr", "images_per_s",
                                          "val_cer", "val_wer", "val_line_acc", "elapsed_s")])
                        log_file.flush()
                    if not quiet:
                        print(f"[val]   step {step} CER={report.cer:.4f} WER={report.wer:.4f} "
                              f"line_acc={report.line_accuracy:.3f} "
                              f"conf={report.mean_confidence:.3f}", flush=True)
                    if report.cer < best["cer"]:
                        best = {"cer": report.cer, "wer": report.wer, "step": step,
                                "report": report.summary()}
                        model_mod.save_checkpoint(
                            checkpoint, model, cfg, fonts_info,
                            metrics={"val_cer": report.cer, "val_wer": report.wer,
                                     "val_line_acc": report.line_accuracy, "step": step},
                        )
                        if not quiet:
                            print(f"[train] new best CER={report.cer:.4f} -> {checkpoint}",
                                  flush=True)
    finally:
        if log_file:
            log_file.close()

    result = {
        "checkpoint": str(checkpoint),
        "steps": step,
        "elapsed_s": time.time() - start,
        "best": best,
        "history": history,
        "fonts": fonts_info,
    }
    if not quiet:
        print(f"[train] done in {result['elapsed_s']/60:.1f} min, best: {best['report']}",
              flush=True)
        print(f"[train] checkpoint: {checkpoint}", flush=True)
    return result


