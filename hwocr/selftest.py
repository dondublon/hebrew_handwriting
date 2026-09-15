"""Fast end-to-end self-test.

Runs the whole pipeline on a tiny scale and checks every stage, including a
short overfitting run that verifies that the model can actually learn
(architecture, CTC wiring, label order, decoding) and that the *inference*
pre-processing matches the training pre-processing.

Usage::

    python -m hwocr.cli selftest          # ~3 minutes
    python -m hwocr.cli selftest --full   # longer overfitting run
"""

from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import (augment, bidi, charset, dataset as ds, decode, detect, fonts, lexicon,
               metrics, model as model_mod, ocr, render)
from .config import Config


class Checks:
    """Collects (name, ok, detail, seconds) records."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str, float]] = []

    def run(self, name: str, fn: Callable[[], str]) -> bool:
        start = time.time()
        try:
            detail = fn() or ""
            ok = True
        except Exception as exc:  # noqa: BLE001 - a self-test must report everything
            detail = f"{type(exc).__name__}: {exc}"
            ok = False
        self.results.append((name, ok, detail, time.time() - start))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} ({time.time() - start:.1f}s) {detail}", flush=True)
        return ok

    @property
    def ok(self) -> bool:
        return all(item[1] for item in self.results)

    def summary(self) -> str:
        passed = sum(1 for item in self.results if item[1])
        return f"{passed}/{len(self.results)} checks passed"
