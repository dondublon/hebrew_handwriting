"""Synthetic datasets.

Training data is generated on the fly: a map-style ``Dataset`` whose content
is a pure function of ``(seed, index)``.  This gives an effectively infinite
training stream while keeping the validation subset bit-for-bit reproducible,
and no disk space is required.

:class:`OnDiskDataset` reads an exported dataset (see :mod:`hwocr.generate`),
which is what the evaluation and the Tesseract baseline use.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import augment, bidi, charset, fonts, lexicon, render
from .config import Config

VAL_SEED = 424242
TEST_SEED = 7331733


class SyntheticLineDataset(Dataset):
    """Random text lines rendered with the registered fonts."""

    def __init__(
        self,
        cfg: Config,
        specs: Sequence[fonts.FontSpec] | None = None,
        weights: np.ndarray | None = None,
        seed: int = 0,
        length: int = 1_000_000,
        heavy_prob: float | None = None,
        words: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.specs = list(specs) if specs else fonts.resolve_fonts(cfg)
        self.weights = (
            weights if weights is not None
            else fonts.sampling_weights(self.specs, cfg.font_weights)
        )
        self.seed = int(seed)
        self.length = int(length)
        self.heavy_prob = cfg.p_heavy_aug if heavy_prob is None else float(heavy_prob)

        base_words = list(words) if words else lexicon.default_lexicon()
        self.prepared: list[tuple[fonts.FontSpec, set[str], list[str]]] = []
        for spec in self.specs:
            allowed = set(spec.supported())
            self.prepared.append((spec, allowed, lexicon.filter_lexicon(base_words, allowed)))

    def __len__(self) -> int:
        return self.length

    def sample_text(self, rng: random.Random) -> tuple[str, str, "fonts.FontSpec"]:
        """Returns (logical text, visual text, font)."""
        idx = int(rng.choices(range(len(self.prepared)), weights=self.weights, k=1)[0])
        spec, allowed, words = self.prepared[idx]
        text = lexicon.sample_text(rng, self.cfg, allowed, words)
        return text, bidi.logical_to_visual(text), spec

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed * 1_000_003 + int(index))
        _, visual, spec = self.sample_text(rng)
        label = charset.encode(visual)
        if not label:  # pathological case - fall back to a fixed sample
            visual, label = "םולש", charset.encode("םולש")

        image = render.render_line(visual, spec, rng, self.cfg)
        image = augment.apply_augment(image, rng, heavy=rng.random() < self.heavy_prob)
        arr, valid_w = augment.to_model_input(image, self.cfg.img_height, self.cfg.max_width)
        return {
            "image": torch.from_numpy(arr),
            "label": torch.tensor(label, dtype=torch.long),
            "valid_width": int(valid_w),
            "font": spec.name,
            "logical_text": bidi.visual_to_logical(visual),
        }


class OnDiskDataset(Dataset):
    """Dataset exported by :mod:`hwocr.generate` (``labels.tsv`` + PNG files)."""

    def __init__(self, root: str | Path, cfg: Config) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.entries: list[tuple[Path, str, str]] = []
        labels = self.root / "labels.tsv"
        if not labels.exists():
            raise FileNotFoundError(f"{labels} not found - run the 'generate' command first")
        for line in labels.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            self.entries.append((self.root / parts[0], parts[1], parts[2]))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        path, visual, logical = self.entries[index]
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        arr, valid_w = augment.to_model_input(image, self.cfg.img_height, self.cfg.max_width)
        return {
            "image": torch.from_numpy(arr),
            "label": torch.tensor(charset.encode(visual), dtype=torch.long),
            "valid_width": int(valid_w),
            "font": path.parent.name,
            "logical_text": logical,
            "path": str(path),
        }


def collate(batch: list[dict], width_reduction: int = 8) -> dict:
    """Pad images/labels of a batch; keeps per-sample CTC lengths."""
    max_w = max(item["image"].shape[-1] for item in batch)
    max_w = int(np.ceil(max_w / width_reduction) * width_reduction)
    max_len = max(item["label"].numel() for item in batch)

    images = torch.full((len(batch), 1, batch[0]["image"].shape[1], max_w), 1.0)
    labels = torch.zeros((len(batch), max_len), dtype=torch.long)
    valid_widths = torch.zeros(len(batch), dtype=torch.long)
    label_lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        w = item["image"].shape[-1]
        images[i, :, :, :w] = item["image"]
        labels[i, :item["label"].numel()] = item["label"]
        valid_widths[i] = item["valid_width"]
        label_lengths[i] = item["label"].numel()
    return {
        "image": images,
        "label": labels,
        "valid_width": valid_widths,
        "label_length": label_lengths,
        "fonts": [item["font"] for item in batch],
        "logical_text": [item["logical_text"] for item in batch],
        "paths": [item.get("path", "") for item in batch],
    }


def ctc_input_lengths(valid_widths: torch.Tensor, width_reduction: int, max_t: int) -> torch.Tensor:
    """Convert pixel widths into CTC time steps (>= 1)."""
    lengths = torch.div(valid_widths, width_reduction, rounding_mode="floor")
    return lengths.clamp(min=1, max=max_t).to(torch.long)


def _worker_init(worker_id: int) -> None:
    # Data generation is CPU heavy; keep OpenCV single threaded inside workers.
    cv2.setNumThreads(0)
    np.random.seed(torch.initial_seed() % (2 ** 32))


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool, workers: int | None = None) -> DataLoader:
    workers = cfg.workers if workers is None else workers
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=lambda batch: collate(batch, cfg.width_reduction),
        worker_init_fn=_worker_init,
        drop_last=shuffle,
        persistent_workers=False,
    )

