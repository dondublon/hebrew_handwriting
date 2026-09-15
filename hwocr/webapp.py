"""Tiny web UI (standard library only).

Serves a single page that can

* upload/scan-drop an image (line detection + recognition), and
* render a Hebrew phrase with a registered font and recognise it again, which
  shows the whole pipeline - including the CER - without preparing any file.

The browser posts the raw image bytes, so no multipart parser is needed.
"""

from __future__ import annotations

import base64
import io
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import augment, charset, dataset as ds, fonts, metrics, ocr, render
from .config import DEFAULT_CHECKPOINT

INDEX_HTML = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<title>hwocr - распознавание рукописного иврита</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f5f7; color: #1c1e21; }
  header { background: #1c2b4a; color: #fff; padding: 14px 22px; }
  header h1 { font-size: 18px; margin: 0; }
  header span { opacity: .75; font-size: 13px; }
  main { max-width: 1000px; margin: 22px auto; padding: 0 16px 60px; }
  section { background: #fff; border-radius: 10px; padding: 18px 20px; margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,.12); }
  h2 { font-size: 15px; margin: 0 0 12px; text-transform: uppercase; letter-spacing: .04em;
       color: #40506b; }
  #drop { border: 2px dashed #9aa7bd; border-radius: 10px; padding: 30px; text-align: center;
          color: #5b6a86; cursor: pointer; transition: .15s; }
  #drop.hover { background: #eef3ff; border-color: #3d64d6; }
  textarea { width: 100%; box-sizing: border-box; font-size: 15px; font-family: inherit;
             padding: 8px 10px; border: 1px solid #c8d0dd; border-radius: 8px; direction: rtl; }
  button { background: #2f5bd4; color: #fff; border: 0; border-radius: 8px; padding: 9px 16px;
           font-size: 14px; cursor: pointer; }
  button:disabled { opacity: .55; cursor: default; }
  pre.out { direction: rtl; unicode-bidi: plaintext; background: #f8fafc; border: 1px solid #dde3ec;
            border-radius: 8px; padding: 12px; font-size: 20px; line-height: 1.6; margin: 0;
            white-space: pre-wrap; min-height: 34px; width: 100%; }
  .meta { color: #55637d; font-size: 13px; margin-top: 8px; }
  img.preview { max-width: 100%; border-radius: 8px; border: 1px solid #dde3ec; }
  .row { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .err { color: #b3261e; }
  code { background: #eef1f6; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<header>
  <h1>hwocr &mdash; OCR рукописного иврита <span id="status"></span></h1>
</header>
<main>
  <section>
    <h2>1. Распознать картинку</h2>
    <div id="drop">Перетащите сюда файл с текстом или кликните, чтобы выбрать<br>
      <span class="meta">PNG / JPG, одна строка или целая страница</span></div>
    <input type="file" id="file" accept="image/*" hidden>
    <div class="row">
      <label class="meta"><input type="checkbox" id="segment" checked> разбивать на строки</label>
    </div>
    <div class="row">
      <pre class="out" id="text-out"></pre>
    </div>
    <div class="meta" id="text-meta"></div>
    <div class="row"><img class="preview" id="annotated" hidden></div>
  </section>

  <section>
    <h2>2. Демо: сгенерировать текст выбранным шрифтом и распознать</h2>
    <textarea id="demo-text" rows="2" placeholder="שלום עולם">החתול הקטן ישן על הספה</textarea>
    <div class="row">
      <button id="demo-run">Сгенерировать и распознать</button>
      <label class="meta"><input type="checkbox" id="demo-heavy"> имитация фото (шум, тень, JPEG)</label>
      <span class="meta" id="demo-meta"></span>
    </div>
    <div class="row"><img class="preview" id="demo-image" hidden></div>
    <div class="row"><pre class="out" id="demo-out"></pre></div>
    <div class="meta" id="demo-score"></div>
  </section>

  <section>
    <h2>Консольный интерфейс</h2>
    <div class="meta">
      <code>python -m hwocr.cli recognize --image file.png</code><br>
      <code>python -m hwocr.cli demo --text "שלום עולם" --out demo.png</code><br>
      <code>python -m hwocr.cli eval</code> &middot;
      <code>python -m hwocr.cli baseline --split data/test_clean</code>
    </div>
  </section>
</main>
<script>
const status = document.getElementById('status');
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const textOut = document.getElementById('text-out');
const textMeta = document.getElementById('text-meta');
const annotated = document.getElementById('annotated');

function showStatus(s) { status.textContent = '| ' + s; }

async function recognize(blob) {
  showStatus('распознаю...');
  const segment = document.getElementById('segment').checked;
  const url = '/api/recognize' + (segment ? '' : '?segment=0');
  const resp = await fetch(url, { method: 'POST', body: blob });
  if (!resp.ok) { showStatus('ошибка'); textMeta.innerHTML =
      '<span class="err">' + (await resp.text()) + '</span>'; return; }
  const data = await resp.json();
  textOut.textContent = data.text || '(пусто)';
  textMeta.textContent = 'строк: ' + data.lines.length + ', средняя уверенность: ' +
      (data.confidence * 100).toFixed(1) + '%, размер: ' + data.image_size.join('x');
  if (data.annotated) { annotated.src = 'data:image/png;base64,' + data.annotated;
      annotated.hidden = false; }
  showStatus('готово');
}

drop.onclick = () => fileInput.click();
fileInput.onchange = () => { if (fileInput.files[0]) recognize(fileInput.files[0]); };
['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => {
  const f = ev.dataTransfer.files[0]; if (f) recognize(f); });

document.getElementById('demo-run').onclick = async () => {
  const button = document.getElementById('demo-run');
  button.disabled = true; showStatus('генерирую...');
  try {
    const resp = await fetch('/api/demo', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: document.getElementById('demo-text').value,
                             heavy: document.getElementById('demo-heavy').checked }) });
    const data = await resp.json();
    document.getElementById('demo-image').src = 'data:image/png;base64,' + data.image;
    document.getElementById('demo-image').hidden = false;
    document.getElementById('demo-out').textContent = data.text || '(пусто)';
    document.getElementById('demo-score').textContent =
      'шрифт: ' + data.font + ' | ожидалось: ' + data.expected +
      ' | CER=' + data.cer.toFixed(4) + ' WER=' + data.wer.toFixed(4) +
      ' | уверенность ' + (data.confidence * 100).toFixed(1) + '%';
    showStatus('готово');
  } catch (e) { showStatus('ошибка: ' + e); }
  finally { button.disabled = false; }
};

fetch('/api/status').then(r => r.json()).then(d => {
  showStatus('модель: ' + d.checkpoint + ', шрифтов: ' + d.fonts.length +
             ', классов: ' + d.classes + ', CER(val): ' + d.val_cer);
});
</script>
</body>
</html>
"""


def _png_base64(image: np.ndarray, max_side: int = 1400) -> str:
    """Encode a (grayscale or BGR) numpy image as a base64 PNG."""
    if max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _annotate(image: np.ndarray, page) -> np.ndarray:
    """Draw the detected line boxes on a copy of the input image."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for line in page.lines:
        if not line.box:
            continue
        x0, y0, x1, y1 = line.box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 220), 2)
    return canvas


class Handler(BaseHTTPRequestHandler):
    """Serves the page plus the /api endpoints."""

    recognizer: ocr.Recognizer | None = None
    lock = threading.Lock()
    server_version = "hwocr/0.1"

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:  # keep the console readable
        print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            cfg = self.recognizer.cfg
            metrics = (self.recognizer.meta.get("metrics") or {})
            val_cer = metrics.get("val_cer")
            self._send_json({
                "checkpoint": Path(DEFAULT_CHECKPOINT).name,
                "fonts": [spec.name for spec in fonts.resolve_fonts(cfg)],
                "classes": charset.num_classes(),
                "val_cer": None if val_cer is None else round(float(val_cer), 4),
            })
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path, _, query = self.path.partition("?")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            if path == "/api/recognize":
                self._recognize(body, segment="segment=0" not in query)
            elif path == "/api/demo":
                self._demo(json.loads(body.decode("utf-8") or "{}"))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            self._send(500, f"{type(exc).__name__}: {exc}".encode("utf-8"), "text/plain")

    # -- endpoints -------------------------------------------------------
    def _recognize(self, body: bytes, segment: bool = True) -> None:
        if not body:
            self._send_json({"error": "empty request body"}, code=400)
            return
        array = np.frombuffer(body, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self._send_json({"error": "unsupported image format"}, code=400)
            return
        with self.lock:  # one inference at a time: the machine has no GPU
            page = self.recognizer.recognize_page(image, segment=segment)
        payload = page.as_dict()
        payload["annotated"] = _png_base64(_annotate(image, page))
        self._send_json(payload)

    def _demo(self, request: dict) -> None:
        cfg = self.recognizer.cfg
        specs = fonts.resolve_fonts(cfg)
        rng = random.Random(int(request.get("seed") or 0) or
                            abs(hash(request.get("text", ""))) % 10000)
        text = (request.get("text") or "").strip()
        if not text:
            sampler = ds.SyntheticLineDataset(cfg, specs, seed=rng.randrange(10 ** 6), length=1)
            text = sampler.sample_text(rng)[0]
        spec = specs[int(rng.choices(range(len(specs)), k=1)[0])]
        image = render.render_text(text, spec, rng, cfg)
        if request.get("heavy"):
            image = augment.apply_heavy(image, rng)

        with self.lock:
            page = self.recognizer.recognize_page(image, segment=True)
        report = metrics.Report()
        report.add(text, page.text)

        self._send_json({
            "expected": text,
            "text": page.text,
            "lines": [line.as_dict() for line in page.lines],
            "confidence": round(page.confidence, 4),
            "cer": round(report.cer, 4),
            "wer": round(report.wer, 4),
            "font": spec.name,
            "image": _png_base64(image),
        })


def serve(checkpoint: str | Path = DEFAULT_CHECKPOINT, host: str = "127.0.0.1",
          port: int = 8009) -> None:
    """Load the checkpoint and serve the UI until interrupted."""
    Handler.recognizer = ocr.Recognizer.load(checkpoint)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[web] hwocr UI on http://{host}:{port}  (checkpoint: {checkpoint})")
    print("[web] Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[web] stopped")
    finally:
        server.server_close()


