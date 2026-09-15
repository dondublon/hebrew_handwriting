"""Isolated-letter dataset: a warm start for the CTC stage.

CTC training from scratch on words is a cold-start problem: the network has to
learn letter shapes *and* the alignment at the same time, which on a CPU budget
converges far too slowly (the model tends to collapse to "all blanks").

Rendering isolated letters is trivial for this project (we own the font
renderer), so the CNN is first trained as a plain 27-class letter classifier.
The weights then transfer directly into the CRNN, because it is the very same
network - see :mod:`hwocr.pretrain`.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import Dataset

from . import augment, charset, fonts, render
from .config import Config

# Letters, final forms and digits - everything except the space.
LETTER_ALPHABET = charset.HEBREW_LETTERS + charset.DIGITS


class SingleLetterDataset(Dataset):
    """Random renderings of a single character (index == ``charset`` index)."""

    def __init__(
        self,
        cfg: Config,
        specs: Sequence[fonts.FontSpec] | None = None,
        seed: int = 0,
        length: int = 200_000,
        augment_prob: float = 0.5,
        chars: str = LETTER_ALPHABET,
    ) -> None:
        self.cfg = cfg
        self.specs = list(specs) if specs else fonts.resolve_fonts(cfg)
        self.weights = fonts.sampling_weights(self.specs, cfg.font_weights)
        self.seed = int(seed)
        self.length = int(length)
        self.augment_prob = float(augment_prob)
        self.chars = "".join(ch for ch in chars if ch in charset.CHAR_TO_IDX)
        self.prepared: list[tuple[fonts.FontSpec, str]] = []
        for spec in self.specs:
            available = "".join(ch for ch in self.chars if ch in spec.supported())
            if available:
                self.prepared.append((spec, available))

    def __len__(self) -> int:
        return self.length

    def _choose(self, index: int) -> tuple[random.Random, "fonts.FontSpec", str]:
        rng = random.Random(self.seed * 1_000_003 + int(index))
        spec, available = self.prepared[
            int(rng.choices(range(len(self.prepared)), weights=self.weights, k=1)[0])
        ]
        return rng, spec, rng.choice(available)

    def letter_for_index(self, index: int) -> str:
        return self._choose(index)[2]

    def __getitem__(self, index: int) -> dict:
        rng, spec, letter = self._choose(index)
        image = render.render_line(letter, spec, rng, self.cfg)
        if rng.random() < self.augment_prob:
            image = augment.apply_augment(image, rng, heavy=rng.random() < self.cfg.p_heavy_aug)
        canvas, valid_w = augment.to_model_input(image, self.cfg.img_height, self.cfg.max_width)
        return {
            "image": torch.from_numpy(canvas),
            "target": charset.CHAR_TO_IDX[letter],
            "letter": letter,
            "valid_width": int(valid_w),
        }
