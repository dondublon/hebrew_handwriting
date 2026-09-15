"""Font registry.

Adding another font to the project is supposed to be as simple as dropping a
``*.ttf`` (or ``*.otf``) file into ``fonts/``: the registry reads the font
name, checks that the font can render Hebrew and exposes it to the synthetic
data generator with a sampling weight.

``font_weights`` in :class:`hwocr.config.Config` can be used to bias the
sampling (e.g. train mostly on one font and only occasionally on another).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from fontTools.ttLib import TTFont
from PIL import ImageFont

from . import charset
from .config import Config

FONT_SUFFIXES = (".ttf", ".otf", ".ttc")


@dataclass(frozen=True)
class FontSpec:
    """A font file that the synthetic generator is allowed to use."""

    path: Path
    family: str
    style: str
    chars: frozenset[str]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        return f"{self.family} {self.style}".strip()

    def supported(self) -> frozenset[str]:
        """Model alphabet characters that this font can actually render."""
        return frozenset(c for c in charset.CHARS if c in self.chars)

    def missing(self) -> list[str]:
        return [c for c in charset.CHARS if c not in self.chars]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.label})"


def _name_record(font: TTFont, name_id: int) -> str:
    try:
        for record in font["name"].names:
            if record.nameID == name_id:
                try:
                    return record.toUnicode()
                except Exception:  # pragma: no cover - broken name tables
                    continue
    except Exception:  # pragma: no cover
        pass
    return ""


@lru_cache(maxsize=256)
def read_font(path: str) -> tuple[str, str, frozenset[str]]:
    """Read (family, style, unicode coverage) from a font file."""
    font = TTFont(path, fontNumber=0, lazy=True)
    family = _name_record(font, 1) or Path(path).stem
    style = _name_record(font, 2) or "Regular"
    chars = frozenset(chr(cp) for cp in font.getBestCmap().keys())
    font.close()
    return family, style, chars


def load_pil_font(path: str | Path, size: int) -> ImageFont.FreeTypeFont:
    """Cached Pillow font loader (rendering is done at a large size and then
    downscaled, so the cache hit rate is high)."""
    return _load_pil_font(str(path), int(size))


@lru_cache(maxsize=128)
def _load_pil_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def font_specs(paths: Iterable[str | Path], require_hebrew: bool = True) -> list[FontSpec]:
    """Build :class:`FontSpec` objects, skipping fonts without Hebrew support."""
    specs: list[FontSpec] = []
    for path in paths:
        path = Path(path)
        try:
            family, style, chars = read_font(str(path))
        except Exception as exc:  # pragma: no cover - unreadable font
            print(f"[fonts] skipping {path.name}: {exc}")
            continue
        spec = FontSpec(path=path, family=family, style=style, chars=chars)
        hebrew_found = len(chars & set(charset.HEBREW_LETTERS))
        if require_hebrew and hebrew_found < 20:
            print(f"[fonts] skipping {path.name}: only {hebrew_found}/27 Hebrew letters")
            continue
        specs.append(spec)
    return specs


def discover_fonts(dirs: Sequence[str | Path], require_hebrew: bool = True) -> list[FontSpec]:
    """Recursively collect all fonts from the given directories."""
    paths: list[Path] = []
    for directory in dirs:
        directory = Path(directory)
        if not directory.exists():
            continue
        if directory.is_file():
            paths.append(directory)
            continue
        for suffix in FONT_SUFFIXES:
            paths.extend(sorted(directory.rglob(f"*{suffix}")))
    return font_specs(paths, require_hebrew=require_hebrew)


def resolve_fonts(
    cfg: Config,
    extra_fonts: Sequence[str | Path] = (),
    extra_dirs: Sequence[str | Path] = (),
    include_system: bool = False,
) -> list[FontSpec]:
    """All fonts used for training/evaluation.

    Order of collection: everything found in ``cfg.font_dirs`` plus optional
    extra fonts/directories, optionally the system font directories.
    """
    dirs: list[str | Path] = list(cfg.font_dirs) + list(extra_dirs)
    if include_system:
        dirs += list(cfg.system_font_dirs)
    specs = discover_fonts(dirs)
    specs += font_specs(extra_fonts)
    # De-duplicate while keeping the order stable.
    seen: set[str] = set()
    unique: list[FontSpec] = []
    for spec in specs:
        key = str(spec.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    if not unique:
        raise RuntimeError(
            "No usable Hebrew fonts found. Put a .ttf file into 'fonts/' or pass --font."
        )
    return unique


def sampling_weights(specs: Sequence[FontSpec], weights: dict[str, float]) -> np.ndarray:
    """Normalised sampling probabilities for a list of fonts."""
    raw = np.array([float(weights.get(spec.name, 1.0)) for spec in specs], dtype=np.float64)
    if raw.sum() <= 0:
        raw = np.ones(len(specs), dtype=np.float64)
    return raw / raw.sum()


def describe(specs: Sequence[FontSpec], weights: dict[str, float] | None = None) -> str:
    """Human readable report about the registered fonts."""
    weights = weights or {}
    lines = []
    for spec in specs:
        missing = spec.missing()
        lines.append(
            f"- {spec.name}: {spec.label} | weight={weights.get(spec.name, 1.0)}"
            f" | unsupported chars: {''.join(missing) if missing else 'none'}"
        )
    return "\n".join(lines)
