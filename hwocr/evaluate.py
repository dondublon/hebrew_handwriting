"""Evaluation of a trained checkpoint.

Two modes are supported:

* exported datasets (``data/<split>``) - the same images are also fed to the
  Tesseract baseline, so the numbers are directly comparable;
* on-the-fly synthetic splits (clean / camera-like).

Misrecognised lines can be dumped to disk with their reference and hypothesis
so that failures are easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from . import augment, charset, dataset as ds, decode, fonts, metrics, model as model_mod, ocr
from .config import Config


@dataclass
class ErrorExample:
    reference: str
    hypothesis: str
    confidence: float
    image: np.ndarray | None = None


@dataclass
class EvalResult:
    name: str
    report: metrics.Report = field(default_factory=metrics.Report)
    errors: list[ErrorExample] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    lexicon_report: metrics.Report | None = None

    def line(self) -> str:
        return f"{self.name:22s} {self.report.summary()}"

    def lexicon_line(self) -> str:
        if self.lexicon_report is None:
            return f"{self.name:22s} (lexicon decoding not evaluated)"
        return f"{self.name:22s} {self.lexicon_report.summary()}  [lexicon-corrected]"


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    cfg: Config,
    loader,
    device: torch.device = torch.device("cpu"),
    name: str = "eval",
    max_errors: int = 0,
) -> EvalResult:
    model.eval()
    result = EvalResult(name=name)
    for batch in loader:
        images = batch["image"].to(device)
        log_probs = model(images)
        lengths = ds.ctc_input_lengths(batch["valid_width"], cfg.width_reduction,
                                       log_probs.shape[0]).tolist()
        decoded = decode.greedy_decode(log_probs.cpu(), lengths)
        for i, (item, reference) in enumerate(zip(decoded, batch["logical_text"])):
            result.report.add(reference, item.logical, item.confidence)
            result.pairs.append((reference, item.logical))
            if item.logical != reference and len(result.errors) < max(1, max_errors):
                image = None
                if max_errors and batch["paths"][i]:
                    path = Path(batch["paths"][i])
                    if path.exists():
                        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
                result.errors.append(ErrorExample(reference, item.logical, item.confidence, image))
    return result


def apply_lexicon_decoding(result: EvalResult,
                           corrector: "postprocess.WordCorrector | None" = None) -> EvalResult:
    """Adds a second report computed on lexicon-corrected hypotheses."""
    from .postprocess import WordCorrector
    from .lexicon import default_lexicon

    corrector = corrector or WordCorrector(default_lexicon())
    report = metrics.Report()
    for reference, hypothesis in result.pairs:
        corrected = corrector.correct_text(hypothesis)
        report.add(reference, corrected)
        if reference != corrected:
            for example in result.errors:
                if example.hypothesis == hypothesis:
                    example.hypothesis = corrected
    result.lexicon_report = report
    return result


def evaluate_split(
    model: torch.nn.Module,
    cfg: Config,
    root: str | Path,
    device: torch.device = torch.device("cpu"),
    max_errors: int = 0,
    lexicon_decode: bool = False,
) -> EvalResult:
    dataset = ds.OnDiskDataset(root, cfg)
    loader = ds.make_loader(dataset, cfg, shuffle=False, workers=0)
    result = evaluate_loader(model, cfg, loader, device, name=Path(root).name,
                             max_errors=max_errors)
    if lexicon_decode:
        apply_lexicon_decoding(result)
    return result


def evaluate_synthetic(
    model: torch.nn.Module,
    cfg: Config,
    specs: Sequence[fonts.FontSpec],
    seed: int,
    samples: int,
    heavy_prob: float,
    device: torch.device = torch.device("cpu"),
    name: str = "synthetic",
) -> EvalResult:
    dataset = ds.SyntheticLineDataset(cfg, specs, seed=seed, length=samples,
                                      heavy_prob=heavy_prob)
    loader = ds.make_loader(dataset, cfg, shuffle=False, workers=0)
    return evaluate_loader(model, cfg, loader, device, name=name)


def save_error_examples(result: EvalResult, out_dir: str | Path, limit: int = 40) -> Path:
    """Write failing lines as PNGs named after the reference text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, example in enumerate(result.errors[:limit]):
        if example.image is None:
            continue
        safe_ref = "".join(ch for ch in example.reference if ch not in '\\/:*?"<>|')[:40]
        safe_hyp = "".join(ch for ch in example.hypothesis if ch not in '\\/:*?"<>|')[:40]
        Image.fromarray(example.image).save(
            out_dir / f"{result.name}_{i:03d}_ref-{safe_ref}_hyp-{safe_hyp}.png"
        )
        saved += 1
    return out_dir


def evaluate_checkpoint(
    checkpoint: str | Path,
    splits: dict[str, str | Path],
    device: str = "cpu",
    max_errors: int = 0,
    error_dir: str | Path | None = None,
    lexicon_decode: bool = False,
) -> dict[str, EvalResult]:
    """Evaluate one checkpoint on several exported splits."""
    model, cfg, _ = model_mod.load_checkpoint(checkpoint, device=device)
    results: dict[str, EvalResult] = {}
    for name, root in splits.items():
        result = evaluate_split(model, cfg, root, torch.device(device), max_errors=max_errors,
                                lexicon_decode=lexicon_decode)
        results[name] = result
        print(f"[eval] {result.line()}", flush=True)
        if lexicon_decode:
            print(f"[eval] {result.lexicon_line()}", flush=True)
        if error_dir and result.errors:
            save_error_examples(result, Path(error_dir) / name)
    return results


def demo_predictions(
    checkpoint: str | Path,
    cfg: Config | None = None,
    n: int = 5,
    heavy_prob: float = 0.0,
    seed: int = 2024,
) -> list[tuple[str, str, float]]:
    """Small qualitative demo: (reference, hypothesis, confidence)."""
    recognizer = ocr.Recognizer.load(checkpoint)
    specs = fonts.resolve_fonts(cfg or recognizer.cfg)
    dataset = ds.SyntheticLineDataset(recognizer.cfg, specs, seed=seed, length=n,
                                      heavy_prob=heavy_prob)
    out = []
    for i in range(n):
        item = dataset[i]
        image = item["image"][0].numpy()
        canvas = (np.clip(image, -1, 1) + 1) * 127.5
        decoded = recognizer.recognize_line(canvas.astype(np.uint8))
        out.append((item["logical_text"], decoded.logical, decoded.confidence))
    return out
