"""Smoke test of the web UI API (run without a browser).

Starts ``hwocr.cli serve`` on a free port, checks the page markup and the
recognition endpoints - including the hand-edited line boxes that the box editor
in the browser sends (``X-Boxes``) and the lexicon switch (``X-Lexicon``).

    python3 scripts/web_smoke_test.py            # uses runs/page_demo.png
    python3 scripts/web_smoke_test.py photo.jpg
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(base: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/status", timeout=5):
                return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def post(base: str, path: str, data: bytes, headers: dict | None = None) -> dict:
    request = urllib.request.Request(base + path, data=data, method="POST",
                                     headers=headers or {})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def get(base: str, path: str) -> str:
    with urllib.request.urlopen(base + path, timeout=60) as response:
        return response.read().decode("utf-8")


def sample_image() -> Path:
    """A page with a few lines - rendered on the fly when needed."""
    path = ROOT / "runs" / "page_demo.png"
    if path.exists():
        return path
    import random

    from PIL import Image

    from hwocr import fonts, render
    from hwocr.config import Config

    cfg = Config()
    spec = fonts.resolve_fonts(cfg)[0]
    lines = ["שלום עולם", "החתול הקטן ישן על הספה", "מים לחם וחלב"]
    page = render.render_page(lines, spec, random.Random(11), cfg, line_gap=42)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page).save(path)
    return path


def main() -> int:
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else sample_image()
    if not image.exists():
        print(f"image not found: {image}")
        return 2
    body = image.read_bytes()

    label_dir = ROOT / "tmp" / "smoke_labels"
    if label_dir.exists():
        for item in label_dir.iterdir():
            item.unlink()
    label_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "hwocr.cli", "serve", "--port", str(port),
         "--label-dir", str(label_dir)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        wait_for_server(base)

        page = get(base, "/")
        for marker in ('id="overlay"', 'id="btn-add"', 'id="btn-del"', 'X-Boxes',
                       'pointerdown', 'preserveAspectRatio', 'id="coords"',
                       'id="btn-copy"', 'id="btn-apply"', 'badge-text', 'applyCoords'):
            assert marker in page, f"the page is missing {marker}"
        print("1. page served, box editor + coordinates panel present")

        auto = post(base, "/api/recognize", body)
        assert auto["auto_detected"] is True and auto["lines"], auto
        print(f"2. auto detection: {len(auto['lines'])} line(s)")
        for line in auto["lines"]:
            print(f"   box={line['box']} text={line['text']!r} conf={line['confidence']:.2f}")

        width, height = auto["image_size"]
        my_boxes = [[20, 40, width - 20, min(height - 1, 120)],
                    [20, min(height - 1, 300), max(40, width - 120),
                     min(height - 1, 380)]]
        manual = post(base, "/api/recognize", body, {"X-Boxes": json.dumps(my_boxes)})
        assert manual["auto_detected"] is False, manual
        assert len(manual["lines"]) == len(my_boxes), manual
        assert manual["lines"][0]["box"] == my_boxes[0], (manual["lines"][0]["box"], my_boxes)
        print("3. hand-edited boxes are used verbatim:")
        for line in manual["lines"]:
            print(f"   box={line['box']} text={line['text']!r}")

        lex = post(base, "/api/recognize", body,
                   {"X-Boxes": json.dumps(my_boxes), "X-Lexicon": "1"})
        print(f"4. lexicon switch works: {[line['text'] for line in lex['lines']]}")

        fallback = post(base, "/api/recognize", body, {"X-Boxes": "[[1,1]]"})
        assert fallback["auto_detected"] is True, fallback
        print("5. unusable boxes fall back to auto detection")

        # 6. разметка: сохраняем вырезанную строку + текст для дообучения
        import base64

        box = auto["lines"][0]["box"]
        correct = "שלום עולם"
        saved = post(base, "/api/label", body, {
            "X-Box": json.dumps(box),
            "X-Label": base64.b64encode(correct.encode("utf-8")).decode("ascii"),
        })
        assert saved["count"] == 1 and saved["text"] == correct, saved
        labels_tsv = label_dir / "labels.tsv"
        rows = labels_tsv.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 and rows[0].startswith(saved["saved"]), rows
        assert (label_dir / saved["saved"]).exists()
        print(f"6. labelling works: {saved['saved']} / {rows[0]!r}")

        fetched = json.load(urllib.request.urlopen(base + "/api/labels", timeout=30))
        assert fetched["count"] == 1, fetched
        print(f"7. /api/labels reports {fetched['count']} sample(s)")

        demo = post(base, "/api/demo",
                    json.dumps({"text": "שלום עולם"}).encode("utf-8"),
                    {"Content-Type": "application/json"})
        assert "expected" in demo and "cer" in demo
        print(f"8. demo endpoint: expected={demo['expected']!r} got={demo['text']!r} "
              f"CER={demo['cer']}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
