"""Synthetic line renderer.

The renderer draws text the way a human hand would: every letter is rendered
into its own tile and then placed with a random rotation, size, baseline
offset and inter-letter spacing.  Together with the image level augmentations
from :mod:`hwocr.augment` this produces training data that is hard enough to
make the recogniser robust, while still being generated from a font.

Because the available Pillow build has no ``libraqm``, text is laid out in
*visual* order (see :mod:`hwocr.bidi`) - the caller passes an already
reordered string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from . import bidi, fonts
from .config import Config

TILE_PAD = 8


@dataclass
class RenderStyle:
    """Knobs that control how "hand written" the result looks."""

    ink_dark: int = 0            # darkest ink value
    ink_light: int = 60          # brightest ink value
    max_rotation: float = 6.0    # degrees, per letter
    max_scale: float = 0.10      # relative letter size jitter
    max_dy: float = 2.0          # px, per letter (at render resolution)
    max_dx: float = 1.2          # px, per letter
    spacing_jitter: float = 0.18  # relative inter-letter spacing jitter
    word_spacing_extra: float = 0.9
    baseline_drift: float = 0.6  # random walk step of the baseline
    baseline_limit: float = 3.5  # px


def _char_tile(font, ch: str, ink: int) -> tuple[Image.Image, int]:
    """Render one character into its own image; returns (tile, ink height)."""
    bbox = font.getbbox(ch)
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    tile = Image.new("L", (width + 2 * TILE_PAD, height + 2 * TILE_PAD), 255)
    draw = ImageDraw.Draw(tile)
    draw.text((TILE_PAD - x0, TILE_PAD - y0), ch, font=font, fill=ink)
    return tile, height


def _transform_tile(tile: Image.Image, rng: random.Random, style: RenderStyle,
                    scale: float, stretch: float) -> Image.Image:
    angle = rng.uniform(-style.max_rotation, style.max_rotation)
    if abs(angle) > 0.2:
        tile = tile.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    width, height = tile.size
    new_size = (max(1, int(round(width * scale * stretch))),
                max(1, int(round(height * scale))))
    if new_size != (width, height):
        tile = tile.resize(new_size, Image.BICUBIC)
    return tile


def measure_body_height(font, text: str) -> float:
    """Median ink height of the letters of ``text`` (used for normalisation).

    Normalising by the *median letter height* instead of the font size makes
    the pipeline independent of the metrics of a particular font, which is
    what allows other fonts to be added later without touching the model.
    """
    heights = []
    for ch in text:
        if ch == " ":
            continue
        bbox = font.getbbox(ch)
        heights.append(max(1, bbox[3] - bbox[1]))
    return float(np.median(heights)) if heights else 1.0


def render_line(
    text_visual: str,
    spec: "fonts.FontSpec",
    rng: random.Random,
    cfg: Config,
    style: RenderStyle | None = None,
) -> np.ndarray:
    """Render a single line of text; returns a uint8 grayscale image (ink dark)."""
    style = style or RenderStyle()
    font = fonts.load_pil_font(spec.path, cfg.font_size)
    ascent, descent = font.getmetrics()

    body = measure_body_height(font, text_visual)
    norm = cfg.target_body / body if body > 0 else 1.0

    margin = 12
    canvas = Image.new(
        "L",
        (max(16, int(font.getlength(text_visual)) + 4 * margin), ascent + descent + 2 * margin),
        255,
    )
    y0 = margin

    cursor = float(margin)
    drift = 0.0
    for ch in text_visual:
        if ch == " ":
            cursor += font.getlength(ch) * rng.uniform(0.8, 1.6) * style.word_spacing_extra
            continue

        ink = rng.randint(style.ink_dark, style.ink_light)
        tile, _ = _char_tile(font, ch, ink)
        drift = float(np.clip(drift + rng.uniform(-style.baseline_drift, style.baseline_drift),
                              -style.baseline_limit, style.baseline_limit))
        scale = norm * (1.0 + rng.uniform(-style.max_scale, style.max_scale))
        stretch = 1.0 + rng.uniform(-0.08, 0.12)
        tile = _transform_tile(tile, rng, style, scale, stretch)

        dx = rng.uniform(-style.max_dx, style.max_dx)
        dy = rng.uniform(-style.max_dy, style.max_dy) + drift
        x = int(round(cursor + dx))
        canvas.paste(ink, (x, int(round(y0 + dy))), Image.fromarray(255 - np.asarray(tile)))

        advance = font.getlength(ch) * (1.0 + rng.uniform(-style.spacing_jitter,
                                                          style.spacing_jitter))
        cursor += advance * norm + dx

    return np.asarray(canvas, dtype=np.uint8)


def render_text(
    logical_text: str,
    spec: "fonts.FontSpec",
    rng: random.Random,
    cfg: Config,
    style: RenderStyle | None = None,
) -> np.ndarray:
    """Render *logical* Hebrew text (typing order) as a single image line."""
    return render_line(bidi.logical_to_visual(logical_text), spec, rng, cfg, style)


def render_page(
    logical_lines: Sequence[str],
    spec: "fonts.FontSpec",
    rng: random.Random,
    cfg: Config,
    line_gap: int = 30,
    margin: int = 40,
) -> np.ndarray:
    """Render several lines onto one page (used by the layout detection demo)."""
    lines = [render_text(line, spec, rng, cfg) for line in logical_lines]
    width = max(line.shape[1] for line in lines) + 2 * margin
    height = sum(line.shape[0] for line in lines) + line_gap * (len(lines) - 1) + 2 * margin
    page = np.full((height, width), 255, dtype=np.uint8)
    y = margin
    for line in lines:
        x = margin + rng.randint(0, max(0, width - 2 * margin - line.shape[1]))
        y1 = min(page.shape[0], y + line.shape[0])
        x1 = min(page.shape[1], x + line.shape[1])
        page[y:y1, x:x1] = np.minimum(page[y:y1, x:x1], line[: y1 - y, : x1 - x])
        y += line.shape[0] + line_gap
    return page

