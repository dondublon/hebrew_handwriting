"""High level inference API (this is what the CLI and the web app use)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import augment, decode, detect, model as model_mod
from .config import DEFAULT_CHECKPOINT, Config


@dataclass
class LineResult:
    """Recognised single line of text."""

    logical: str
    visual: str
    confidence: float
    box: list[int] | None = None
    font_hint: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.logical,
            "visual": self.visual,
            "confidence": round(self.confidence, 4),
            "box": self.box,
        }


@dataclass
class PageResult:
    """Recognised page (one or more lines)."""

    lines: list[LineResult] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    checkpoint: str = ""

    @property
    def text(self) -> str:
        return "\n".join(line.logical for line in self.lines)

    @property
    def confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "image_size": list(self.image_size),
            "checkpoint": self.checkpoint,
            "lines": [line.as_dict() for line in self.lines],
        }


class Recognizer:
    """Wraps the trained CRNN for convenient inference."""

    def __init__(self, model: model_mod.CRNN, cfg: Config, meta: dict | None = None,
                 device: str = "cpu") -> None:
        self.model = model
        self.cfg = cfg
        self.meta = meta or {}
        self.device = device

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu") -> "Recognizer":
        model, cfg, meta = model_mod.load_checkpoint(checkpoint, device=device)
        return cls(model, cfg, meta, device=device)

    # ------------------------------------------------------------------
    def recognize_batch(self, images: list[np.ndarray]) -> list[decode.Decoded]:
        """Recognise a batch of already cropped text lines."""
        if not images:
            return []
        prepared = []
        widths = []
        for image in images:
            canvas, valid_w = augment.prepare_for_model(
                image, self.cfg.img_height, self.cfg.max_width
            )
            prepared.append(canvas)
            widths.append(valid_w)
        max_w = max(canvas.shape[1] for canvas in prepared)
        batch = np.full((len(prepared), 1, self.cfg.img_height, max_w), 255, dtype=np.uint8)
        for i, canvas in enumerate(prepared):
            batch[i, 0, :, :canvas.shape[1]] = canvas
        tensor = torch.from_numpy(augment.normalize(batch)).to(self.device)

        with torch.no_grad():
            log_probs = self.model(tensor)
        lengths = [
            self.model.feature_lengths(min(w, max_w)) for w in widths
        ]
        return decode.greedy_decode(log_probs.cpu(), lengths)

    def recognize_line(self, image: np.ndarray) -> decode.Decoded:
        return self.recognize_batch([image])[0]

    # ------------------------------------------------------------------
    def recognize_page(self, image: np.ndarray | str | Path, segment: bool = True) -> PageResult:
        """Recognise a full page (line detection + recognition)."""
        if isinstance(image, (str, Path)):
            image = np.asarray(Image.open(image).convert("L"), dtype=np.uint8)
        gray = augment.ensure_gray(image)
        if not segment:
            crops = [gray]
            boxes: list[detect.LineBox | None] = [None]
        else:
            boxes = detect.segment_lines(gray)
            if not boxes:
                boxes = [detect.LineBox(0, 0, gray.shape[1], gray.shape[0])]
            crops = [box.crop(gray) for box in boxes]

        decoded = self.recognize_batch(crops)
        lines = [
            LineResult(
                logical=item.logical,
                visual=item.visual,
                confidence=item.confidence,
                box=box.as_list() if box is not None else None,
            )
            for item, box in zip(decoded, boxes)
        ]
        return PageResult(lines=lines, image_size=(gray.shape[1], gray.shape[0]))
