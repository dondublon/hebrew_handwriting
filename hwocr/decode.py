"""CTC decoding and text metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from . import bidi, charset


@dataclass
class Decoded:
    """Result of decoding one text line."""

    visual: str
    logical: str
    confidence: float
    token_confidences: list[float] = field(default_factory=list)
    token_frames: list[int] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.logical


def greedy_decode(log_probs: torch.Tensor, input_lengths: Sequence[int]) -> list[Decoded]:
    """Standard CTC greedy (best-path) decoding.

    ``log_probs`` has shape (T, B, C).  Returned text is in *visual* order plus
    the matching *logical* (Hebrew reading) order.
    """
    probs = log_probs.exp()
    best_idx = probs.argmax(dim=2)        # (T, B)
    best_prob = probs.max(dim=2).values   # (T, B)
    batch = log_probs.shape[1]
    results: list[Decoded] = []

    for b in range(batch):
        length = int(input_lengths[b]) if input_lengths is not None else log_probs.shape[0]
        length = max(0, min(length, log_probs.shape[0]))
        chars: list[str] = []
        confidences: list[float] = []
        frames: list[int] = []
        prev = charset.BLANK_INDEX
        for t in range(length):
            idx = int(best_idx[t, b])
            if idx == prev:
                continue
            prev = idx
            if idx == charset.BLANK_INDEX:
                continue
            ch = charset.IDX_TO_CHAR.get(idx)
            if ch is None:
                continue
            chars.append(ch)
            confidences.append(float(best_prob[t, b]))
            frames.append(t)
        visual = "".join(chars)
        results.append(
            Decoded(
                visual=visual,
                logical=bidi.visual_to_logical(visual),
                confidence=sum(confidences) / len(confidences) if confidences else 0.0,
                token_confidences=confidences,
                token_frames=frames,
            )
        )
    return results
