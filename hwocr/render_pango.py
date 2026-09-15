"""High-fidelity rendering through Pango / HarfBuzz.

The in-process renderer (:mod:`hwocr.render`) uses Pillow, whose build here has
no ``libraqm``: no shaping, no bidi and no kerning.  ``pango-view`` *does* have
all three, so it is used to produce an independent test set: text rendered the
way a word processor would render it.

Consequences of this cross-check:

* the spatial layout of the Pango render is the ground truth for
  :mod:`hwocr.bidi` (if our visual-order conversion disagreed with Pango's
  bidi, the CER on this split would collapse);
* it also validates that the model works on images it was never trained on
  (different renderer, different rasteriser, real kerning).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from . import augment, fonts


def pango_available() -> bool:
    return shutil.which("pango-view") is not None


@contextmanager
def _fontconfig(font_dirs: list[Path]):
    """Temporary fontconfig setup so Pango can see fonts that are not installed."""
    with tempfile.TemporaryDirectory(prefix="hwocr_fc_") as tmp:
        tmp_path = Path(tmp)
        config = tmp_path / "fonts.conf"
        cache = tmp_path / "cache"
        cache.mkdir(exist_ok=True)
        entries = "\n".join(f"  <dir>{d}</dir>" for d in font_dirs)
        config.write_text(
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
            "<fontconfig>\n"
            f"{entries}\n"
            f"  <cachedir>{cache}</cachedir>\n"
            "</fontconfig>\n",
            encoding="utf-8",
        )
        previous = os.environ.get("FONTCONFIG_FILE")
        os.environ["FONTCONFIG_FILE"] = str(config)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("FONTCONFIG_FILE", None)
            else:
                os.environ["FONTCONFIG_FILE"] = previous


def render_text(
    text_logical: str,
    spec: "fonts.FontSpec",
    size: int = 44,
    margin: int = 8,
    timeout: float = 20.0,
) -> np.ndarray:
    """Render *logical* Hebrew text with Pango; returns uint8 grayscale (ink dark)."""
    if not pango_available():
        raise RuntimeError("pango-view not found in PATH")
    with tempfile.TemporaryDirectory(prefix="hwocr_pango_") as tmp:
        output = Path(tmp) / "line.png"
        with _fontconfig([spec.path.parent]):
            result = subprocess.run(
                [
                    "pango-view", "--no-display",
                    "--font", f"{spec.family} {size}",
                    "--text", text_logical,
                    "--background", "white",
                    "--foreground", "black",
                    "--margin", str(margin),
                    "--output", str(output),
                ],
                capture_output=True, text=True, timeout=timeout,
            )
        if result.returncode != 0 or not output.exists():
            raise RuntimeError(f"pango-view failed: {result.stderr.strip()[:200]}")
        image = np.asarray(Image.open(output).convert("L"), dtype=np.uint8)
    return augment.ensure_gray(image)


def render_batch(
    texts: list[str],
    spec: "fonts.FontSpec",
    size: int = 44,
    margin: int = 8,
    progress_every: int = 50,
) -> list[np.ndarray]:
    """Render several lines (one ``pango-view`` call each - see the module docstring)."""
    images: list[np.ndarray] = []
    for i, text in enumerate(texts):
        images.append(render_text(text, spec, size=size, margin=margin))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[pango] {i + 1}/{len(texts)}", flush=True)
    return images
