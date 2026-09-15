"""Command line interface: the entry point of the whole proof of concept.

Examples::

    python -m hwocr.cli list-fonts
    python -m hwocr.cli selftest
    python -m hwocr.cli generate
    python -m hwocr.cli train
    python -m hwocr.cli eval
    python -m hwocr.cli demo --text "שלום עולם"
    python -m hwocr.cli recognize --image page.png
    python -m hwocr.cli baseline --split data/test_clean --limit 100
    python -m hwocr.cli serve --port 8009
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_CHECKPOINT, DATA_DIR, RUNS_DIR, Config
from . import pretrain as pretrain_mod


def _build_config(args: argparse.Namespace) -> Config:
    """Config from a JSON file (if given) plus the most useful CLI overrides."""
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    overrides = {
        "batch_size": getattr(args, "batch_size", None),
        "lr": getattr(args, "lr", None),
        "steps_short": getattr(args, "steps_short", None),
        "steps_medium": getattr(args, "steps_medium", None),
        "steps_light": getattr(args, "steps_light", None),
        "steps_heavy": getattr(args, "steps_heavy", None),
        "val_every": getattr(args, "val_every", None),
        "val_samples": getattr(args, "val_samples", None),
        "workers": getattr(args, "workers", None),
        "torch_threads": getattr(args, "torch_threads", None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    extra_dirs = getattr(args, "font_dir", None)
    if extra_dirs:
        cfg.font_dirs = list(cfg.font_dirs) + [str(d) for d in extra_dirs]
    return cfg


def _load_words(args: argparse.Namespace) -> list[str] | None:
    from . import lexicon as lexicon_mod

    path = getattr(args, "lexicon", None)
    if not path:
        return None
    words = lexicon_mod.load_user_lexicon(path)
    print(f"[cli] lexicon: {len(words)} words from {path}")
    return words


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_list_fonts(args: argparse.Namespace) -> int:
    from . import charset, fonts

    cfg = _build_config(args)
    specs = fonts.resolve_fonts(cfg, extra_fonts=args.font or [],
                               extra_dirs=args.font_dir or [],
                               include_system=args.include_system_fonts)
    print(f"[cli] {len(specs)} font(s) available:")
    print(fonts.describe(specs, cfg.font_weights))
    print(f"\n[cli] alphabet: {len(charset.CHARS)} characters "
          f"({charset.num_classes()} classes incl. CTC blank)")
    print(f"[cli] characters: {' '.join(charset.CHARS)}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from . import selftest

    cfg = _build_config(args)
    ok = selftest.run_all(cfg, quick=args.quick, overfit_steps=args.overfit_steps,
                          overfit_samples=args.overfit_samples)
    return 0 if ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    from . import dataset as ds, fonts, train

    cfg = _build_config(args)
    words = _load_words(args)
    specs = fonts.resolve_fonts(cfg, extra_fonts=args.font or [],
                               extra_dirs=args.font_dir or [],
                               include_system=args.include_system_fonts)
    print("[cli] registered fonts:\n" + fonts.describe(specs, cfg.font_weights))
    result = train.train(
        cfg=cfg,
        checkpoint=args.checkpoint,
        extra_fonts=args.font or [],
        extra_dirs=args.font_dir or [],
        include_system=args.include_system_fonts,
        words=words,
        log_path=args.log or (RUNS_DIR / "train_log.csv"),
        init_from=args.init_from,
    )
    print(f"[cli] best: {result['best']['report']} at step {result['best']['step']}")
    print(f"[cli] next: python -m hwocr.cli eval --checkpoint {result['checkpoint']}")
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from . import fonts, pretrain

    cfg = _build_config(args)
    specs = fonts.resolve_fonts(cfg, extra_fonts=args.font or [],
                               extra_dirs=args.font_dir or [],
                               include_system=args.include_system_fonts)
    result = pretrain.train_letters(cfg, specs, steps=args.steps,
                                    batch_size=args.batch_size or 32,
                                    lr=args.lr or 2e-3, out_path=args.out,
                                    val_every=args.val_every)
    print(f"[cli] letter accuracy {result['accuracy']:.4f} -> {result['path']}")
    print(f"[cli] next: python -m hwocr.cli train --init-from {result['path']}")
    return 0


def _discover_splits(root: Path) -> dict[str, Path]:
    splits = {}
    if root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "labels.tsv").exists():
                splits[child.name] = child
    return splits


def cmd_eval(args: argparse.Namespace) -> int:
    from . import evaluate

    if args.split:
        splits = {Path(item).name: Path(item) for item in args.split}
    else:
        splits = _discover_splits(Path(args.data_dir))
    if not splits:
        print("[cli] no exported splits found - run 'generate' first")
        return 2
    results = evaluate.evaluate_checkpoint(
        args.checkpoint, splits, max_errors=args.save_errors, error_dir=args.errors_dir,
        lexicon_decode=args.lexicon_decode,
    )
    print("\n[cli] summary")
    for name, result in results.items():
        print(f"  {result.line()}")
        if args.lexicon_decode:
            print(f"  {result.lexicon_line()}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from . import dataset as ds, fonts, generate

    cfg = _build_config(args)
    words = _load_words(args)
    specs = fonts.resolve_fonts(cfg, extra_fonts=args.font or [],
                               extra_dirs=args.font_dir or [],
                               include_system=args.include_system_fonts)

    sizes = {"sample": args.sample, "val": args.val, "test_clean": args.test_clean,
             "test_camera": args.test_camera}
    extra_roots: dict[str, list[str]] = {}
    if args.extra_font_test_dir:
        from pathlib import Path as _Path
        paths = [str(p) for d in args.extra_font_test_dir
                 for p in _Path(d).rglob("*.ttf")]
        if paths:
            extra_roots["other"] = paths

    result = generate.generate_all(cfg, specs, words=words, sizes=sizes,
                                   root=args.out, extra_font_roots=extra_roots,
                                   pango_size=args.pango, pango_size_px=args.pango_size)
    for name, path in result.items():
        print(f"[cli] {name:20s} -> {path}")
    print(f"[cli] hint: validation seed is {ds.VAL_SEED}, test seed {ds.TEST_SEED}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Render Hebrew text with a registered font and recognise it again."""
    import random

    from . import augment, dataset as ds, fonts, metrics, ocr, render

    recognizer = ocr.Recognizer.load(args.checkpoint)
    cfg = recognizer.cfg
    specs = fonts.resolve_fonts(cfg, extra_fonts=args.font or [],
                               extra_dirs=args.font_dir or [],
                               include_system=args.include_system_fonts)
    rng = random.Random(args.seed)
    weights = fonts.sampling_weights(specs, cfg.font_weights)
    sampler = ds.SyntheticLineDataset(cfg, specs, seed=args.seed, length=1, heavy_prob=0.0)

    lines = []
    if args.text:
        lines.append(args.text)
    while len(lines) < max(1, args.lines):
        lines.append(sampler.sample_text(rng)[0])

    spec = specs[int(rng.choices(range(len(specs)), weights=weights, k=1)[0])]
    if len(lines) == 1:
        image = render.render_text(lines[0], spec, rng, cfg)
    else:
        image = render.render_page(lines, spec, rng, cfg)

    heavy = rng.random() < args.heavy
    if heavy:
        image = augment.apply_heavy(image, rng)
        if len(lines) > 1:  # a photo is not perfectly straight
            image = augment.random_rotate(image, rng, 1.5)

    out_path = args.out
    if out_path:
        from PIL import Image as _Image
        _Image.fromarray(image).save(out_path)

    page = recognizer.recognize_page(image, segment=not args.no_segment)
    expected = "\n".join(lines)
    report = metrics.Report()
    predicted_lines = [line.logical for line in page.lines]
    if len(predicted_lines) == len(lines):
        for reference, hypothesis in zip(lines, predicted_lines):
            report.add(reference, hypothesis)
    else:
        report.add(" ".join(lines), " ".join(predicted_lines))

    print(f"[demo] font:      {spec.label} ({spec.name})")
    print(f"[demo] photo-like augmentation: {'yes' if heavy else 'no'}")
    if out_path:
        print(f"[demo] image:     {out_path}")
    print(f"[demo] lines detected: {len(page.lines)}")
    print(f"[demo] expected:  {expected.replace(chr(10), ' | ')}")
    print(f"[demo] recognised:{' ' + ' | '.join(predicted_lines)}")
    print(f"[demo] CER={report.cer:.4f} WER={report.wer:.4f} "
          f"confidence={page.confidence:.3f}")
    if args.json:
        print(json.dumps(page.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.cer == 0 else 1


def _annotate(image, page) -> "object":
    """Draw the detected line boxes (and the text, if a Hebrew font is found)."""
    from PIL import Image as _Image, ImageDraw, ImageFont

    canvas = _Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = None
    for candidate in ("/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, 18)
            break
        except OSError:
            continue
    for line in page.lines:
        if not line.box:
            continue
        x0, y0, x1, y1 = line.box
        draw.rectangle([x0, y0, x1, y1], outline=(220, 40, 40), width=2)
        if font:
            from . import bidi
            draw.text((x0, max(0, y0 - 20)), bidi.logical_to_visual(line.logical),
                      font=font, fill=(20, 90, 200))
    return canvas


def cmd_recognize(args: argparse.Namespace) -> int:
    from . import ocr

    recognizer = ocr.Recognizer.load(args.checkpoint)
    page = recognizer.recognize_page(args.image, segment=not args.no_segment)
    if args.json:
        print(json.dumps(page.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(page.text)
        print(f"\n[cli] {len(page.lines)} line(s), mean confidence "
              f"{page.confidence:.3f}, checkpoint {Path(args.checkpoint).name}")
    if args.out_image:
        import numpy as np
        from PIL import Image as _Image
        annotated = _annotate(np.asarray(_Image.open(args.image).convert("L")), page)
        annotated.save(args.out_image)
        print(f"[cli] annotated image: {args.out_image}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Compare Tesseract with our model on the very same exported split."""
    from . import baseline_tesseract as baseline, evaluate

    if not baseline.tesseract_available():
        print("[cli] tesseract is not installed")
        return 2
    if not args.split:
        print("[cli] --split is required (e.g. --split data/test_clean)")
        return 2
    root = Path(args.split)
    print(f"[cli] tesseract langs: {', '.join(baseline.available_langs()[:12])} ...")
    print(f"[cli] split: {root}")

    tess = baseline.evaluate_split(root, lang=args.lang, psm=args.psm, limit=args.limit)
    print(f"[baseline] tesseract (psm={args.psm}) {tess.summary()}")

    if Path(args.checkpoint).exists():
        from . import evaluate, model as model_mod

        model, cfg, _ = model_mod.load_checkpoint(args.checkpoint)
        result = evaluate.evaluate_split(model, cfg, root, max_errors=0)
        print(f"[baseline] hwocr CRNN          {result.report.summary()}")
        improvement = (tess.cer - result.report.cer) / tess.cer * 100 if tess.cer else 0.0
        print(f"[baseline] relative CER improvement over tesseract: {improvement:.1f}%")
    else:
        print(f"[baseline] checkpoint {args.checkpoint} not found - train first")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import webapp

    webapp.serve(args.checkpoint, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def _add_font_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--font", action="append", metavar="PATH",
                        help="extra font file to use (repeatable)")
    parser.add_argument("--font-dir", action="append", metavar="DIR",
                        help="extra directory with fonts (repeatable)")
    parser.add_argument("--include-system-fonts", action="store_true",
                        help="also scan the system font directories")


def _add_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps-plain", type=int,
                        help="stage 1 steps (clean renderings, short texts)")
    parser.add_argument("--steps-light", type=int, help="stage 2 steps (mild augmentations)")
    parser.add_argument("--steps-heavy", type=int, help="stage 3 steps (photo-like)")
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--lr", type=float, help="peak learning rate")
    parser.add_argument("--workers", type=int, help="DataLoader workers")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--lexicon", help="plain text word list to use instead of the built-in one")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hwocr.cli",
        description="OCR proof of concept for Hebrew computer fonts that imitate handwriting",
    )
    parser.add_argument("--config", help="JSON config file (see hwocr/config.py)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-fonts", help="show the font registry")
    _add_font_args(p)
    p.set_defaults(func=cmd_list_fonts)

    p = sub.add_parser("selftest", help="fast end-to-end self-test")
    _add_tuning_args(p)
    p.add_argument("--quick", action="store_true", help="skip the (slower) learning check")
    p.add_argument("--overfit-steps", type=int, default=80,
                   help="training steps for the learning check")
    p.add_argument("--overfit-samples", type=int, default=24)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("generate", help="export synthetic datasets to data/")
    _add_font_args(p)
    _add_tuning_args(p)
    p.add_argument("--out", help="output root (default: data/)")
    p.add_argument("--sample", type=int, default=500, help="size of the 'sample' split")
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--test-clean", type=int, default=500)
    p.add_argument("--test-camera", type=int, default=500)
    p.add_argument("--extra-font-test-dir", action="append", metavar="DIR",
                   help="directory with additional fonts for a zero-shot font test")
    p.add_argument("--pango", type=int, default=0,
                   help="also export a Pango/HarfBuzz-rendered test split of this size")
    p.add_argument("--pango-size", type=int, default=44, help="font size for the Pango split")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="train the CRNN")
    _add_font_args(p)
    _add_tuning_args(p)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--log", help="CSV log path (default: runs/train_log.csv)")
    p.add_argument("--init-from", help="warm start from a 'pretrain' checkpoint")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("pretrain", help="warm start: train the CNN on isolated letters")
    _add_font_args(p)
    _add_tuning_args(p)
    p.add_argument("--out", default=str(pretrain_mod.DEFAULT_PRETRAINED))
    p.add_argument("--steps", type=int, default=800)
    p.set_defaults(func=cmd_pretrain)

    p = sub.add_parser("eval", help="evaluate a checkpoint on exported splits")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--split", action="append", help="exported split directory (repeatable)")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--save-errors", type=int, default=0,
                   help="keep at most this many misrecognised lines per split")
    p.add_argument("--errors-dir", help="where to write the error examples")
    p.add_argument("--lexicon-decode", action="store_true",
                   help="also score the output corrected with the word list")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("demo", help="render text with the font and recognise it")
    _add_font_args(p)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--text", help="text to render (default: a random sample)")
    p.add_argument("--lines", type=int, default=1, help="number of lines to render")
    p.add_argument("--heavy", type=float, default=0.0,
                   help="probability of photo-like augmentations (0..1)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", help="save the rendered image here")
    p.add_argument("--no-segment", action="store_true", help="treat the image as one line")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("recognize", help="recognise text on an image file")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--no-segment", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out-image", help="save an annotated copy of the input")
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("baseline", help="compare with Tesseract on the same data")
    p.add_argument("--split", required=True)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--lang", default="heb")
    p.add_argument("--psm", type=int, default=7)
    p.add_argument("--limit", type=int, help="evaluate only the first N lines")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("serve", help="start the small web UI")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8009)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[cli] interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())




