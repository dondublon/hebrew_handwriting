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
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import bidi, charset, fonts
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
    word_spacing_extra: float = 0.9
    baseline_drift: float = 0.6  # random walk step of the baseline
    baseline_limit: float = 3.5  # px
    side_margin: int = 12        # white margin around the line, px
    # Both extremes of the font's own spacing are useful training data: the
    # connected look is authentic, the separated one is unambiguous.
    p_natural_spacing: float = 0.5   # keep the font's (possibly touching) spacing
    min_gap: float = 1.5             # px, enforced when letters are pushed apart



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
    """Median ink height of the letters of ``text`` (debug helper)."""
    heights = []
    for ch in text:
        if ch == " ":
            continue
        bbox = font.getbbox(ch)
        heights.append(max(1, bbox[3] - bbox[1]))
    return float(np.median(heights)) if heights else 1.0


@lru_cache(maxsize=64)
def font_body_height(path: str, size: int) -> float:
    """Median ink height of the Hebrew alphabet at a given font size.

    The normalisation factor is calibrated *per font*, not per rendered line:
    a per-line median depends on which letters happen to be in the string
    (tall letters such as ל raise the median, flat ones such as ם lower it),
    which made the width of a line vary by ~40% for the same number of
    characters.  A per-font calibration keeps the letter size consistent, and
    makes the image width predictable from the text length (used for batch
    bucketing).
    """
    font = fonts.load_pil_font(path, size)
    heights = []
    for ch in charset.HEBREW_LETTERS:
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
    """Render a single line of text; returns a uint8 grayscale image (ink dark).

    Layout is done in *font units* using the font's own advances and its GPOS
    kerning - i.e. the way a real word processor would set this text.  Every
    glyph is rendered into its own tile and then placed by the **centre of its
    ink**, which keeps the placement correct even though the tiles are randomly
    rotated and scaled.  (Placing tiles by a pen cursor instead shifts each
    letter by its side bearing, which for this italic font is up to 13 px -
    the letters then drift apart or collide at random.)
    """
    style = style or RenderStyle()
    font = fonts.load_pil_font(spec.path, cfg.font_size)
    ascent, descent = font.getmetrics()
    kerning = fonts.kerning_pairs(str(spec.path))

    body = font_body_height(str(spec.path), cfg.font_size)
    # Per-line size jitter keeps the samples varied while the calibration
    # itself stays font specific.
    norm = cfg.target_body / body if body > 0 else 1.0
    norm *= 1.0 + rng.uniform(-cfg.line_scale_jitter, cfg.line_scale_jitter)
    if norm <= 0:
        norm = 1.0

    # ---- layout pass: pen positions in font units ----------------------
    glyphs: list[tuple[str, float, float]] = []      # (char, ink centre x/y)
    pen = style.side_margin / norm
    previous_ink_right: float | None = None
    previous_char: str | None = None
    for ch in text_visual:
        if ch == " ":
            pen += font.getlength(ch) * rng.uniform(0.8, 1.6) * style.word_spacing_extra
            previous_ink_right = None
            previous_char = None
            continue
        x0, y0, x1, y1 = font.getbbox(ch)
        if previous_char is not None:
            pen += kerning.get((previous_char, ch), 0.0)
        # The font is a connected (cursive) style: with its natural advances
        # neighbouring letters can touch or overlap, which is authentic but
        # often ambiguous.  A part of the letters is therefore pushed apart.
        if previous_ink_right is not None and rng.random() > style.p_natural_spacing:
            pen = max(pen, previous_ink_right + style.min_gap / norm - x0)
        glyphs.append((ch, pen + (x0 + x1) / 2.0, (y0 + y1) / 2.0 - ascent))
        previous_ink_right = pen + x1
        previous_char = ch
        pen += font.getlength(ch)

    # ---- paint ---------------------------------------------------------
    width = int(round(pen * norm)) + 2 * style.side_margin
    height = int(round((ascent + descent) * norm)) + 2 * style.side_margin
    canvas = Image.new("L", (max(16, width), max(12, height)), 255)
    baseline = style.side_margin + ascent * norm

    drift = 0.0
    for ch, centre_x, centre_y in glyphs:
        ink = rng.randint(style.ink_dark, style.ink_light)
        tile, _ = _char_tile(font, ch, ink)
        drift = float(np.clip(drift + rng.uniform(-style.baseline_drift, style.baseline_drift),
                              -style.baseline_limit, style.baseline_limit))
        scale = norm * (1.0 + rng.uniform(-style.max_scale, style.max_scale))
        stretch = 1.0 + rng.uniform(-0.08, 0.12)
        tile = _transform_tile(tile, rng, style, scale, stretch)
        ink_box = ImageOps.invert(tile).getbbox()
        if ink_box is None:
            continue
        dx = rng.uniform(-style.max_dx, style.max_dx)
        dy = rng.uniform(-style.max_dy, style.max_dy) + drift
        target_x = centre_x * norm + dx
        target_y = baseline + centre_y * norm + dy
        position = (
            int(round(target_x - (ink_box[0] + ink_box[2]) / 2.0)),
            int(round(target_y - (ink_box[1] + ink_box[3]) / 2.0)),
        )
        canvas.paste(ink, position, Image.fromarray(255 - np.asarray(tile)))

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

