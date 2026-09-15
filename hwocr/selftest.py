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


def _check_charset_and_fonts(cfg: Config) -> str:
    specs = fonts.resolve_fonts(cfg)
    missing = {spec.name: "".join(spec.missing()) for spec in specs}
    bad = {name: chars for name, chars in missing.items() if chars}
    if bad:
        raise AssertionError(f"fonts missing requested characters: {bad}")
    if charset.num_classes() < 30:
        raise AssertionError("alphabet looks too small")
    return (f"{len(specs)} font(s), {charset.num_classes()} classes, "
            f"max label {cfg.max_label_len} chars")


def _check_lexicon() -> str:
    words = lexicon.default_lexicon()
    problems = charset.validate_lexicon(words)
    if problems:
        raise AssertionError(f"{len(problems)} lexicon problems, e.g. {problems[:3]}")
    if len(words) < 200:
        raise AssertionError(f"lexicon too small: {len(words)}")
    return f"{len(words)} words ({len(set(words))} unique)"


def _check_bidi(cfg: Config, n: int = 2000) -> str:
    rng = random.Random(1)
    specs = fonts.resolve_fonts(cfg)
    allowed = set(specs[0].supported())
    words = lexicon.filter_lexicon(lexicon.default_lexicon(), allowed)
    samples = [lexicon.sample_text(rng, cfg, allowed, words, 20) for _ in range(n)]
    samples += ["שלום 2024", "  שלום", "שלום  ", "(שלום)", "12:30", "א"]
    failures = []
    for text in samples:
        visual = bidi.logical_to_visual(text)
        if bidi.visual_to_logical(visual) != text:
            failures.append((text, visual, bidi.visual_to_logical(visual)))
    if failures:
        raise AssertionError(f"{len(failures)} non-invertible strings, e.g. {failures[:2]}")
    return f"{len(samples)} strings round-tripped"


def _check_dataset(cfg: Config, n: int = 120, min_feasible: float = 0.9) -> str:
    specs = fonts.resolve_fonts(cfg)
    dataset = ds.SyntheticLineDataset(cfg, specs, seed=3, length=n, heavy_prob=cfg.p_heavy_aug)
    feasible = 0
    per_char = []
    for i in range(n):
        item = dataset[i]
        label_len = int(item["label"].numel())
        valid_w = int(item["valid_width"])
        if valid_w <= 0:
            continue
        frames = max(1, valid_w // cfg.width_reduction)
        if 2 * label_len + 1 <= frames:
            feasible += 1
        per_char.append(valid_w / max(1, label_len))
        if item["image"].shape != (1, cfg.img_height, item["image"].shape[-1]):
            raise AssertionError(f"unexpected image shape {tuple(item['image'].shape)}")
        if float(item["image"].min()) < -1.01 or float(item["image"].max()) > 1.01:
            raise AssertionError("normalisation out of range")
    ratio = feasible / n
    if ratio < min_feasible:
        raise AssertionError(
            f"only {ratio:.0%} of the samples fit into the CTC frame budget "
            f"(median {np.median(per_char):.1f} px per character, "
            f"width reduction {cfg.width_reduction})"
        )
    return (f"{ratio:.0%} CTC-feasible, {np.median(per_char):.1f} px/char, "
            f"input height {cfg.img_height}")


def _check_model(cfg: Config, batch: int = 4) -> str:
    model = model_mod.build_model(cfg)
    width = cfg.width_reduction * 8
    x = torch.randn(batch, 1, cfg.img_height, width)
    with torch.no_grad():
        log_probs = model(x)
    expected = (width // cfg.width_reduction, batch, charset.num_classes())
    if tuple(log_probs.shape) != expected:
        raise AssertionError(f"unexpected output shape {tuple(log_probs.shape)} != {expected}")
    if not torch.isfinite(log_probs).all():
        raise AssertionError("non-finite log-probabilities")
    return (f"{model_mod.count_parameters(model)/1e6:.2f}M params, "
            f"output {tuple(log_probs.shape)}")


def _check_checkpoint(cfg: Config) -> str:
    model = model_mod.build_model(cfg)
    model.eval()  # BatchNorm/dropout must be in inference mode for a fair comparison
    x = torch.randn(2, 1, cfg.img_height, cfg.width_reduction * 6)
    with torch.no_grad():
        reference = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model_mod.save_checkpoint(path, model, cfg, "dummy")
        loaded, loaded_cfg, meta = model_mod.load_checkpoint(path)
        with torch.no_grad():
            again = loaded(x)
        if not torch.allclose(reference, again, atol=1e-5):
            raise AssertionError("forward pass differs after save/load")
        if loaded_cfg.img_height != cfg.img_height or meta.get("charset") != charset.CHARS:
            raise AssertionError("configuration/characters not restored")
    return "saved and loaded, outputs identical"


def _check_detect(cfg: Config) -> str:
    specs = fonts.resolve_fonts(cfg)
    rng = random.Random(5)
    lines = ["שלום עולם", "החתול הקטן ישן על הספה", "בדיקת כיול 2024", "מים לחם וחלב"]
    page = render.render_page(lines, specs[0], rng, cfg, line_gap=34)
    boxes = detect.segment_lines(page)
    if len(boxes) != len(lines):
        raise AssertionError(f"detected {len(boxes)} lines instead of {len(lines)}")
    heights = [box.height for box in boxes]
    if min(heights) < 15:
        raise AssertionError(f"line boxes too small: {heights}")
    return f"{len(boxes)} lines detected on a {page.shape[1]}x{page.shape[0]} page"


def _check_learns(cfg: Config, samples: int = 32, steps: int = 200,
                  target_cer: float = 0.10) -> str:
    """Short overfitting run: proves the model can learn *and* that inference works."""
    specs = fonts.resolve_fonts(cfg)
    dataset = ds.SyntheticLineDataset(cfg, specs, seed=909, length=samples, heavy_prob=0.0)
    batch = ds.collate([dataset[i] for i in range(samples)], cfg.width_reduction)

    torch.manual_seed(0)
    model = model_mod.build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = torch.nn.CTCLoss(blank=charset.BLANK_INDEX, zero_infinity=True)
    for _ in range(steps):
        log_probs = model(batch["image"])
        lengths = ds.ctc_input_lengths(batch["valid_width"], cfg.width_reduction,
                                       log_probs.shape[0])
        loss = criterion(log_probs, batch["label"], lengths, batch["label_length"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        log_probs = model(batch["image"])
    decoded = decode.greedy_decode(log_probs, lengths.tolist())
    train_report = metrics.Report()
    for item, reference in zip(decoded, batch["logical_text"]):
        train_report.add(reference, item.logical, item.confidence)
    if train_report.cer > target_cer:
        raise AssertionError(
            f"memorisation failed: CER {train_report.cer:.3f} > {target_cer:.2f} "
            f"after {steps} steps (first prediction: {decoded[0].logical!r} vs "
            f"{batch['logical_text'][0]!r})"
        )

    # Same samples through the *inference* API: catches any mismatch between the
    # training pre-processing and the pre-processing used at prediction time.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "overfit.pt"
        model_mod.save_checkpoint(path, model, cfg, "selftest")
        recognizer = ocr.Recognizer.load(path)
        images = [
            ((np.clip(image[0].numpy(), -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
            for image in batch["image"]
        ]
        predictions = recognizer.recognize_batch(images)
    infer_report = metrics.Report()
    for item, reference in zip(predictions, batch["logical_text"]):
        infer_report.add(reference, item.logical, item.confidence)
    if infer_report.cer > target_cer + 0.05:
        raise AssertionError(
            f"inference path differs from training path: CER {infer_report.cer:.3f} "
            f"vs {train_report.cer:.3f}"
        )
    return (f"train CER {train_report.cer:.3f}, inference CER {infer_report.cer:.3f} "
            f"({samples} samples, {steps} steps)")


def run_all(cfg: Config | None = None, quick: bool = False, overfit_steps: int = 200,
            overfit_samples: int = 32) -> bool:
    """Run every check; returns True when all of them pass."""
    cfg = cfg or Config()
    torch.set_num_threads(cfg.torch_threads)
    checks = Checks()
    print(f"[selftest] hwocr self-test, checkpoint config: "
          f"height={cfg.img_height} width<={cfg.max_width} "
          f"reduction={cfg.width_reduction}", flush=True)

    checks.run("charset/fonts", lambda: _check_charset_and_fonts(cfg))
    checks.run("lexicon", _check_lexicon)
    checks.run("bidi round-trip", lambda: _check_bidi(cfg))
    checks.run("synthetic dataset", lambda: _check_dataset(cfg))
    checks.run("model shapes", lambda: _check_model(cfg))
    checks.run("checkpoint round-trip", lambda: _check_checkpoint(cfg))
    checks.run("line detection", lambda: _check_detect(cfg))
    if not quick:
        checks.run("learning + inference",
                   lambda: _check_learns(cfg, overfit_samples, overfit_steps))
    print(f"[selftest] {checks.summary()}", flush=True)
    return checks.ok


