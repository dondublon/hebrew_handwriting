"""Export synthetic datasets to disk (PNG images + ``labels.tsv``).

On-the-fly generation is used for training, but evaluation, the Tesseract
baseline and error analysis all need a *fixed* dataset that both systems see
byte for byte.  This module produces those datasets.

Layout of an exported split::

    data/test_clean/
        labels.tsv          # "<file>\t<visual text>\t<logical text>"
        <font>_000000.png
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from PIL import Image

from . import augment, bidi, charset, dataset as ds, fonts, lexicon, render
from .config import Config

SPLIT_SEEDS = {
    "sample": (11, 0.0),
    "val": (ds.VAL_SEED, None),
    "test_clean": (ds.TEST_SEED, 0.0),
    "test_camera": (ds.TEST_SEED + 1, 1.0),
}


def _prepare(
    cfg: Config,
    specs: Sequence[fonts.FontSpec],
    words: Sequence[str] | None,
) -> list[tuple[fonts.FontSpec, set[str], list[str]]]:
    base_words = list(words) if words else lexicon.default_lexicon()
    prepared = []
    for spec in specs:
        allowed = set(spec.supported())
        prepared.append((spec, allowed, lexicon.filter_lexicon(base_words, allowed)))
    return prepared


def export_split(
    root: str | Path,
    cfg: Config,
    specs: Sequence[fonts.FontSpec],
    size: int,
    seed: int,
    heavy_prob: float = 0.0,
    words: Sequence[str] | None = None,
    progress_every: int = 250,
) -> Path:
    """Render ``size`` text lines into ``root``; returns the directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    weights = fonts.sampling_weights(specs, cfg.font_weights)
    prepared = _prepare(cfg, specs, words)

    rows: list[str] = []
    for i in range(size):
        spec, allowed, lex = prepared[int(rng.choices(range(len(prepared)), weights=weights, k=1)[0])]
        text = lexicon.sample_text(rng, cfg, allowed, lex)
        visual = bidi.logical_to_visual(text)
        if not charset.is_supported(visual):
            continue
        image = render.render_line(visual, spec, rng, cfg)
        image = augment.apply_augment(image, rng, heavy=rng.random() < heavy_prob)
        name = f"{spec.path.stem.replace(' ', '_')}_{i:06d}.png"
        Image.fromarray(image).save(root / name)
        rows.append(f"{name}\t{visual}\t{text}")
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[generate] {root.name}: {i + 1}/{size}", flush=True)

    (root / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "info.txt").write_text(
        f"seed={seed}\nsize={len(rows)}\nheavy_prob={heavy_prob}\n"
        f"fonts={[spec.name for spec in specs]}\n",
        encoding="utf-8",
    )
    print(f"[generate] {root}: {len(rows)} lines written", flush=True)
    return root


def generate_all(
    cfg: Config | None = None,
    specs: Sequence[fonts.FontSpec] | None = None,
    words: Sequence[str] | None = None,
    sizes: dict[str, int] | None = None,
    root: str | Path | None = None,
    extra_font_roots: dict[str, Sequence[str]] | None = None,
) -> dict[str, Path]:
    """Create the standard splits: sample/val/test_clean/test_camera (+ per-font tests)."""
    cfg = cfg or Config()
    specs = list(specs) if specs else fonts.resolve_fonts(cfg)
    sizes = sizes or {"sample": 500, "val": cfg.val_samples, "test_clean": 500,
                      "test_camera": 500}
    base = Path(root) if root else Path(cfg.data_dir)

    result: dict[str, Path] = {}
    for name, size in sizes.items():
        seed, heavy = SPLIT_SEEDS.get(name, (cfg.seed, 0.0))
        heavy = cfg.p_heavy_aug if heavy is None else heavy
        result[name] = export_split(base / name, cfg, specs, size, seed, heavy, words)

    # One split per font: shows how well the model generalises to a font it
    # has never seen (including fonts added later).
    for name, font_paths in (extra_font_roots or {}).items():
        font_specs = fonts.font_specs(font_paths)
        if not font_specs:
            continue
        result[f"test_font_{name}"] = export_split(
            base / f"test_font_{name}", cfg, font_specs, sizes.get("test_clean", 500),
            ds.TEST_SEED + 2, heavy_prob=0.0, words=words
        )
    return result
