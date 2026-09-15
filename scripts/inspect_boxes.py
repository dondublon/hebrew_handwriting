"""Inspect specific boxes of an image: crop, recognised text, confidence.

Built for the debugging loop "send a file + the coordinates of the boxes":
paste the coordinates exactly as the web UI prints them (or as JSON) and the
script reports, per box, what our model and Tesseract see, and writes the crops
to a folder so they can be looked at.

    python3 scripts/inspect_boxes.py photo.jpeg "1: 151, 46, 309, 86"
    python3 scripts/inspect_boxes.py photo.jpeg --boxes-file boxes.txt
    python3 scripts/inspect_boxes.py photo.jpeg --json "[[151,46,309,86]]"
    python3 scripts/inspect_boxes.py photo.jpeg --whole      # авто-разметка целиком

Options:
    --checkpoint PATH   модель (по умолчанию models/crnn_gveret.pt)
    --out DIR           куда сложить кропы (по умолчанию tmp/inspect)
    --lexicon           включить словарную коррекцию
    --no-tesseract      не сравнивать с Tesseract
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hwocr import baseline_tesseract, detect, ocr
from hwocr.config import DEFAULT_CHECKPOINT


def parse_boxes(text: str) -> list[list[int]]:
    """Coordinates from the web UI: 'N: x0, y0, x1, y1' lines or a JSON array."""
    text = text.strip()
    if text.startswith("["):
        try:
            data = json.loads(text)
            boxes = [[int(round(float(v))) for v in item[:4]]
                     for item in data if isinstance(item, list) and len(item) >= 4]
            if boxes:
                return boxes
        except json.JSONDecodeError:
            pass
    boxes = []
    for line in text.splitlines():
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", line)]
        if len(numbers) >= 4:
            values = numbers[-4:] if len(numbers) >= 5 else numbers[:4]
            boxes.append([int(round(v)) for v in values])
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("boxes", nargs="*", help="координаты: 'x0,y0,x1,y1' или 'N: x0,y0,x1,y1'")
    parser.add_argument("--boxes-file", help="файл с координатами (по строке на рамку)")
    parser.add_argument("--json", help="координаты как JSON [[x0,y0,x1,y1], ...]")
    parser.add_argument("--whole", action="store_true", help="игнорировать рамки, взять авто-разметку")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default="tmp/inspect")
    parser.add_argument("--lexicon", action="store_true")
    parser.add_argument("--no-tesseract", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"файл не найден: {image_path}")
        return 2
    gray = np.asarray(ImageOps.exif_transpose(Image.open(image_path)).convert("L"),
                      dtype=np.uint8)
    print(f"image: {image_path} {gray.shape[1]}x{gray.shape[0]} "
          f"mean={gray.mean():.0f}")

    raw = args.json or ("\n".join(args.boxes) if args.boxes else None)
    if args.boxes_file:
        raw = Path(args.boxes_file).read_text(encoding="utf-8")
    boxes = parse_boxes(raw) if raw else None

    recognizer = ocr.Recognizer.load(args.checkpoint, lexicon_decode=args.lexicon)
    page = recognizer.recognize_page(gray, segment=boxes is None, boxes=boxes)
    if boxes is None:
        print(f"auto detection: {len(page.lines)} box(es)")
    else:
        print(f"given boxes: {len(boxes)}, used: {len(page.lines)}")
    print(f"model: {Path(args.checkpoint).name}, lexicon={args.lexicon}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tesseract_ready = (not args.no_tesseract) and baseline_tesseract.tesseract_available()

    for i, line in enumerate(page.lines, start=1):
        box = line.box or [0, 0, gray.shape[1], gray.shape[0]]
        x0, y0, x1, y1 = box
        crop = gray[max(0, y0):y1, max(0, x0):x1]
        crop_path = out_dir / f"box_{i:02d}.png"
        if crop.size:
            Image.fromarray(crop).save(crop_path)
        # увеличенный вариант — удобно смотреть глазами
        if crop.size:
            big = Image.fromarray(crop)
            scale = max(1, int(round(60 / max(1, crop.shape[0]))))
            if scale > 1:
                big = big.resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.LANCZOS)
            big.save(out_dir / f"box_{i:02d}_x{scale}.png")
        tess = baseline_tesseract.recognize(crop, lang="heb", psm=7) if tesseract_ready else "-"
        density = float((detect.binarize(crop) > 0).mean()) if crop.size else 0.0
        print(f"\n[{i}] box={box} {x1 - x0}x{y1 - y0} ink_density={density:.2f} "
              f"crop={crop_path}")
        print(f"    hwocr : {line.logical!r}  (conf {line.confidence:.2f})")
        print(f"    tess  : {tess!r}")

    print(f"\ncrops: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
