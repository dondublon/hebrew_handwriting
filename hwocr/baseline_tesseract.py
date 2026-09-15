"""Tesseract baseline.

Tesseract 5.2 with the ``heb`` model is available on this machine, but it was
trained on printed Hebrew (mostly ``David``/``Arial``-like shapes) and fails on
handwriting-style fonts.  This module makes that failure measurable instead of
anecdotal: the same exported dataset is fed to Tesseract and to our model, and
both are scored with the same CER/WER code.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import metrics

DEFAULT_PSM = 7  # treat the image as a single text line


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def available_langs() -> list[str]:
    if not tesseract_available():
        return []
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                             text=True, check=False).stdout
    except OSError:  # pragma: no cover
        return []
    return [line.strip() for line in out.splitlines()[1:] if line.strip()]


def recognize(
    image: np.ndarray | str | Path,
    lang: str = "heb",
    psm: int = DEFAULT_PSM,
) -> str:
    """Run Tesseract on one image line; returns the (logical order) text."""
    if not tesseract_available():
        raise RuntimeError("tesseract executable not found in PATH")
    tmpdir = Path(tempfile.mkdtemp(prefix="hwocr_tess_"))
    try:
        if isinstance(image, (str, Path)):
            path = Path(image)
        else:
            path = tmpdir / "line.png"
            Image.fromarray(image).save(path)
        result = subprocess.run(
            ["tesseract", str(path), "-", "-l", lang, "--psm", str(psm)],
            capture_output=True, text=True, check=False,
        )
        return " ".join(result.stdout.split())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def evaluate_split(
    root: str | Path,
    lang: str = "heb",
    psm: int = DEFAULT_PSM,
    limit: int | None = None,
) -> metrics.Report:
    """CER/WER of Tesseract on an exported split (``data/<name>``)."""
    root = Path(root)
    report = metrics.Report()
    entries = [line.split("\t") for line in
               (root / "labels.tsv").read_text(encoding="utf-8").splitlines() if line]
    for i, parts in enumerate(entries):
        if limit is not None and i >= limit:
            break
        name, _, logical = parts[0], parts[1], parts[2]
        hypothesis = recognize(root / name, lang=lang, psm=psm)
        report.add(logical, hypothesis)
    return report


def sweep_psm(root: str | Path, lang: str = "heb", psms: tuple[int, ...] = (6, 7, 13),
              limit: int | None = None) -> dict[int, metrics.Report]:
    """Try a few page-segmentation modes, because Tesseract is very sensitive to it."""
    return {psm: evaluate_split(root, lang=lang, psm=psm, limit=limit) for psm in psms}
