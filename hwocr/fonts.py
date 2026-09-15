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


def _x_advance(value) -> float:
    """XAdvance of a GPOS ValueRecord (0 when absent)."""
    if value is None:
        return 0.0
    advance = getattr(value, "XAdvance", None)
    return float(advance) if advance else 0.0


@lru_cache(maxsize=64)
def kerning_pairs(path: str) -> dict[tuple[str, str], float]:
    """Character pair kerning (in font units) taken from the font's GPOS table.

    The Pillow build used here has no HarfBuzz, so without this the synthetic
    renderings would ignore the font's kerning (up to ±140 units for this
    font), while a real word processor *does* apply it - i.e. the training data
    would not match the images the model is meant to read.
    """
    try:
        font = TTFont(path, lazy=True)
        gpos = font["GPOS"].table
    except Exception:  # pragma: no cover - font without GPOS
        return {}

    cmap = font.getBestCmap()
    glyph_to_char: dict[str, str] = {}
    for codepoint, glyph_name in cmap.items():
        glyph_to_char.setdefault(glyph_name, chr(codepoint))

    pairs: dict[tuple[str, str], float] = {}

    def add(first_glyph: str, second_glyph: str, value: float) -> None:
        first, second = glyph_to_char.get(first_glyph), glyph_to_char.get(second_glyph)
        if first and second and value:
            pairs[(first, second)] = value

    for lookup in getattr(gpos.LookupList, "Lookup", []):
        for subtable in lookup.SubTable:
            subtable = getattr(subtable, "ExtSubTable", subtable)
            if hasattr(subtable, "PairSet") and subtable.Coverage is not None:
                for index, first_glyph in enumerate(subtable.Coverage.glyphs):
                    for record in subtable.PairSet[index].PairValueRecord:
                        add(first_glyph, record.SecondGlyph, _x_advance(record.Value1))
            elif hasattr(subtable, "Class1Record") and subtable.ClassDef1 is not None:
                first_class = _glyphs_by_class(subtable.ClassDef1, glyph_to_char)
                second_class = _glyphs_by_class(subtable.ClassDef2, glyph_to_char)
                for index_1, record_1 in enumerate(subtable.Class1Record):
                    first_glyphs = first_class.get(index_1, [])
                    if not first_glyphs:
                        continue
                    for index_2, record_2 in enumerate(record_1.Class2Record):
                        value = _x_advance(record_2.Value1)
                        if not value:
                            continue
                        for first_glyph in first_glyphs:
                            for second_glyph in second_class.get(index_2, []):
                                add(first_glyph, second_glyph, value)
    return pairs


def _glyphs_by_class(class_def, known_glyphs: dict[str, str]) -> dict[int, list[str]]:
    """Expand a GPOS ClassDef into {class index: [glyph names]}.

    Glyphs that are not mentioned in ``classDefs`` belong to class 0, which is
    why the known glyphs are added explicitly.
    """
    mapping: dict[int, list[str]] = {}
    for glyph, klass in class_def.classDefs.items():
        mapping.setdefault(int(klass), []).append(glyph)
    unassigned = [g for g in known_glyphs if g not in class_def.classDefs]
    if unassigned:
        mapping.setdefault(0, []).extend(unassigned)
    return mapping


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
