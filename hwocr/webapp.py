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
from PIL import Image, ImageOps

from . import augment, bidi, charset, dataset as ds, fonts, metrics, ocr, render
from .config import DEFAULT_CHECKPOINT, PROJECT_ROOT

INDEX_HTML = r"""<!DOCTYPE html>
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
  /* --- редактор рамок -------------------------------------------------- */
  .toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
  button.ghost { background: #e7ebf3; color: #26354d; }
  button.danger { background: #c62828; }
  button.active { outline: 2px solid #2f5bd4; outline-offset: 1px; }
  .editor { position: relative; margin-top: 12px; line-height: 0; user-select: none;
            touch-action: none; border: 1px solid #dde3ec; border-radius: 8px; overflow: hidden; }
  .editor img { display: block; width: 100%; height: auto; }
  .editor svg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  .editor svg .box { fill: rgba(47,91,212,.12); stroke: #2f5bd4; stroke-width: 2; cursor: move; }
  .editor svg .box.selected { fill: rgba(211,47,47,.16); stroke: #d32f2f; }
  .editor svg .handle { fill: #fff; stroke: #2f5bd4; stroke-width: 2; cursor: nwse-resize; }
  .editor svg .handle.selected { stroke: #d32f2f; }
  .editor svg .handle[data-h="n"], .editor svg .handle[data-h="s"] { cursor: ns-resize; }
  .editor svg .handle[data-h="w"], .editor svg .handle[data-h="e"] { cursor: ew-resize; }
  .editor svg .handle[data-h="ne"], .editor svg .handle[data-h="sw"] { cursor: nesw-resize; }
  /* номера рамок */
  .editor svg .badge { fill: #2f5bd4; cursor: move; }
  .editor svg .badge.selected { fill: #d32f2f; }
  .editor svg .badge-text { fill: #fff; font-family: system-ui, sans-serif; pointer-events: none; }
  textarea.coords { width: 100%; box-sizing: border-box; margin-top: 8px; font-family: ui-monospace,
      SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.45; direction: ltr;
      padding: 8px 10px; border: 1px solid #c8d0dd; border-radius: 8px; background: #f8fafc;
      resize: vertical; }
  .results .coord { color: #55637d; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
      monospace; font-size: 12px; cursor: copy; white-space: nowrap; }
  .results .coord:hover { color: #2f5bd4; text-decoration: underline; }
  .results { list-style: none; padding: 0; margin: 12px 0 0; }
  .results li { display: flex; gap: 10px; align-items: baseline; padding: 6px 8px;
                border-radius: 6px; cursor: pointer; border: 1px solid transparent; }
  .results li:hover { background: #f1f4f9; }
  .results li.selected { background: #eef3ff; border-color: #2f5bd4; }
  .results .num { color: #7b879c; font-size: 12px; min-width: 28px; }
  .results .txt { direction: rtl; unicode-bidi: plaintext; font-size: 18px; flex: 1; }
  .results .conf { color: #55637d; font-size: 12px; white-space: nowrap; }
  .results input.label { direction: rtl; unicode-bidi: plaintext; font-family: inherit;
                         font-size: 16px; flex: 1; min-width: 120px; padding: 4px 8px;
                         border: 1px solid #c8d0dd; border-radius: 6px; background: #fff; }
  .results button.save { padding: 4px 9px; font-size: 13px; }
  .results li.saved { background: #eefbf1; }
  .results li.saved .num { color: #2e7d32; }
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
      <span class="meta">PNG / JPG; одна строка или целая страница</span></div>
    <input type="file" id="file" accept="image/*" hidden>
    <div class="toolbar" id="toolbar" hidden>
      <button id="btn-recognize">Распознать рамки</button>
      <button class="ghost" id="btn-auto">Авто-разметка заново</button>
      <button class="ghost" id="btn-add">+ Рамка</button>
      <button class="ghost" id="btn-dup">Дублировать</button>
      <button class="danger" id="btn-del">Удалить</button>
      <button class="ghost" id="btn-clear">Убрать все</button>
      <label class="meta"><input type="checkbox" id="lexicon"> словарь</label>
      <span class="meta" id="label-count"></span>
      <span class="meta">рамки: тяните за уголки/стороны, «+ Рамка» — рисуйте новую; Delete — удалить выбранную</span>
    </div>
    <div class="editor" id="editor" hidden>
      <img id="photo" alt="загруженная картинка">
      <svg id="overlay" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <ul class="results" id="results"></ul>
    <div class="toolbar" id="coords-tools" hidden>
      <button class="ghost" id="btn-copy">Скопировать координаты</button>
      <button class="ghost" id="btn-json">Скопировать JSON</button>
      <button class="ghost" id="btn-apply">Применить координаты из текста</button>
      <span class="meta">формат: «номер: x0, y0, x1, y1» (или JSON) — можно править и вставлять</span>
    </div>
    <textarea class="coords" id="coords" rows="4" spellcheck="false" hidden
      placeholder="1: 100, 50, 300, 90&#10;2: 100, 120, 300, 160"></textarea>
    <div class="row"><pre class="out" id="text-out"></pre></div>
    <div class="meta" id="text-meta"></div>
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
// ---- элементы --------------------------------------------------------------
const status = document.getElementById('status');
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const editor = document.getElementById('editor');
const photo = document.getElementById('photo');
const overlay = document.getElementById('overlay');
const toolbar = document.getElementById('toolbar');
const resultsBox = document.getElementById('results');
const lexiconBox = document.getElementById('lexicon');

// ---- состояние -------------------------------------------------------------
const state = {
  blob: null,      // загруженный файл (переотправляется при изменении рамок)
  natW: 0, natH: 0,
  boxes: [],       // [[x0, y0, x1, y1], ...] в пикселях изображения
  texts: [],       // [{text, confidence}, ...] по рамкам
  labels: [],      // правильный текст, который вводит пользователь
  saved: [],       // сохранён ли пример (для обучающей выборки)
  selected: -1,
  addMode: false,
  busy: false,
};
let drag = null;

function showStatus(s) { status.textContent = '| ' + s; }

function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// UTF-8 -> base64 (иначе иврит не помещается в HTTP-заголовок)
function b64utf8(value) {
  return btoa(unescape(encodeURIComponent(value)));
}

// размер ручек в пикселях изображения, чтобы на экране они были ~10 px
function handleSize() {
  const width = overlay.clientWidth || 1;
  return Math.max(4, (state.natW / width) * 5);
}

function toImage(evt) {
  const r = overlay.getBoundingClientRect();
  return [
    (evt.clientX - r.left) / (r.width || 1) * state.natW,
    (evt.clientY - r.top) / (r.height || 1) * state.natH,
  ];
}

// ---- отрисовка -------------------------------------------------------------
function render() {
  const hs = handleSize();
  overlay.setAttribute('viewBox', '0 0 ' + state.natW + ' ' + state.natH);
  overlay.setAttribute('preserveAspectRatio', 'none');
  let html = '';
  state.boxes.forEach((b, i) => {
    const selected = i === state.selected;
    const cls = 'box' + (selected ? ' selected' : '');
    const x = Math.min(b[0], b[2]), y = Math.min(b[1], b[3]);
    const w = Math.abs(b[2] - b[0]), h = Math.abs(b[3] - b[1]);
    html += '<rect class="' + cls + '" data-idx="' + i + '" x="' + x + '" y="' + y +
            '" width="' + w + '" height="' + h + '" vector-effect="non-scaling-stroke"/>';
    // номер рамки (виден всегда, за него тоже можно тянуть)
    const badgeH = hs * 2.8;
    const badgeW = Math.max(badgeH, badgeH * 0.62 * String(i + 1).length + hs * 0.6);
    const badgeY = y - badgeH >= 0 ? y - badgeH : y;
    html += '<rect class="badge' + (selected ? ' selected' : '') + '" data-idx="' + i +
            '" x="' + x + '" y="' + badgeY + '" width="' + badgeW + '" height="' + badgeH +
            '" rx="' + (hs * 0.6) + '" vector-effect="non-scaling-stroke"/>' +
            '<text class="badge-text" x="' + (x + badgeW / 2) + '" y="' +
            (badgeY + badgeH * 0.78) + '" font-size="' + (badgeH * 0.72) +
            '" text-anchor="middle">' + (i + 1) + '</text>';
    if (selected) {
      const cx = x + w / 2, cy = y + h / 2;
      const points = [[x, y, 'nw'], [cx, y, 'n'], [x + w, y, 'ne'], [x + w, cy, 'e'],
                      [x + w, y + h, 'se'], [cx, y + h, 's'], [x, y + h, 'sw'], [x, cy, 'w']];
      for (const [px, py, name] of points) {
        html += '<rect class="handle selected" data-idx="' + i + '" data-h="' + name +
                '" x="' + (px - hs) + '" y="' + (py - hs) + '" width="' + (hs * 2) +
                '" height="' + (hs * 2) + '" vector-effect="non-scaling-stroke"/>';
      }
    }
  });
  overlay.innerHTML = html;
  renderResults();
  updateCoords();
}

// ---- координаты рамок как текст --------------------------------------------
function coordsLines() {
  return state.boxes.map((b, i) =>
    (i + 1) + ': ' + b.map(v => Math.round(v)).join(', '));
}

function updateCoords() {
  const area = document.getElementById('coords');
  if (document.activeElement !== area) area.value = coordsLines().join('\n');
}

function copyText(text, label) {
  const done = () => showStatus(label + ' скопировано в буфер');
  const fallback = () => {
    const area = document.getElementById('coords');
    area.value = text;
    area.hidden = false;
    area.focus();
    area.select();
    try { document.execCommand('copy'); done(); }
    catch (e) { showStatus('не удалось скопировать — выделите текст вручную'); }
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(fallback);
  } else {
    fallback();
  }
}

function parseCoords(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith('[')) {          // JSON вида [[x0,y0,x1,y1], ...]
    try {
      const data = JSON.parse(trimmed);
      const boxes = [];
      for (const item of (Array.isArray(data) ? data : [])) {
        if (Array.isArray(item) && item.length >= 4) {
          boxes.push(item.slice(0, 4).map(Number));
        }
      }
      if (boxes.length) return boxes;
    } catch (e) { /* пробуем разобрать как текст */ }
  }
  const boxes = [];
  for (const line of trimmed.split(/\n+/)) {
    const numbers = (line.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
    if (numbers.length >= 4) {
      // в строке «3: 10, 20, 30, 40» первое число — номер рамки
      boxes.push(numbers.length >= 5 ? numbers.slice(-4) : numbers.slice(0, 4));
    }
  }
  return boxes;
}

function applyCoords() {
  const boxes = parseCoords(document.getElementById('coords').value);
  if (!boxes.length) { showStatus('в тексте не нашёл координат'); return; }
  state.boxes = boxes.map(b => [b[0], b[1], b[2], b[3]]);
  state.texts = state.boxes.map(() => ({ text: '', confidence: null }));
  state.labels = state.boxes.map(() => '');
  state.saved = state.boxes.map(() => false);
  state.selected = -1;
  render();
  showStatus('применено рамок: ' + boxes.length + ' — нажмите «Распознать рамки»');
}

function renderResults() {
  resultsBox.innerHTML = state.boxes.map((b, i) => {
    const info = state.texts[i] || { text: '', confidence: null };
    const conf = (info.confidence === null || info.confidence === undefined)
      ? '' : (info.confidence * 100).toFixed(0) + '%';
    const typed = state.labels[i] !== undefined ? state.labels[i] : (info.text || '');
    return '<li class="' + (i === state.selected ? 'selected' : '') +
             (state.saved[i] ? ' saved' : '') + '" data-idx="' + i + '">' +
           '<span class="num">#' + (i + 1) + (state.saved[i] ? ' ✓' : '') + '</span>' +
           '<input class="label" data-idx="' + i + '" value="' + escapeHtml(typed) + '"' +
           ' placeholder="правильный текст этой строки">' +
           '<span class="coord" data-idx="' + i + '" title="скопировать координаты этой рамки">[' +
             b.map(v => Math.round(v)).join(',') + ']</span>' +
           '<span class="conf">' + conf + '</span>' +
           '<button class="save" data-idx="' + i + '"' +
           ' title="сохранить вырезанную строку и текст в обучающую выборку">💾</button>' +
           '</li>';
  }).join('');
}

// ---- сохранение обучающих примеров -----------------------------------------
async function saveLabel(index) {
  const text = (state.labels[index] || '').trim();
  const box = state.boxes[index];
  if (!box) return;
  if (!text) { showStatus('введите правильный текст для этой рамки'); return; }
  showStatus('сохраняю пример...');
  try {
    const resp = await fetch('/api/label', {
      method: 'POST',
      headers: {
        'X-Box': JSON.stringify(box.map(v => Math.round(v))),
        'X-Label': b64utf8(text),
      },
      body: state.blob,
    });
    const data = await resp.json();
    if (!resp.ok) { showStatus('ошибка: ' + (data.error || resp.status)); return; }
    state.saved[index] = true;
    document.getElementById('label-count').textContent =
      'примеров размечено: ' + data.count + ' → ' + data.dir;
    render();
    showStatus('пример сохранён: ' + data.saved);
  } catch (e) {
    showStatus('ошибка: ' + e);
  }
}

async function refreshLabelCount() {
  try {
    const resp = await fetch('/api/labels');
    const data = await resp.json();
    document.getElementById('label-count').textContent =
      'примеров размечено: ' + data.count + ' → ' + data.dir;
  } catch (e) { /* не критично */ }
}

// ---- работа с сервером -----------------------------------------------------
async function recognize(auto) {
  if (!state.blob || state.busy) return;
  state.busy = true;
  showStatus(auto ? 'авто-разметка...' : 'распознаю...');
  const headers = { 'X-Lexicon': lexiconBox.checked ? '1' : '0' };
  if (!auto && state.boxes.length) {
    headers['X-Boxes'] = JSON.stringify(state.boxes.map(b => b.map(v => Math.round(v))));
  }
  try {
    const resp = await fetch('/api/recognize', { method: 'POST', headers, body: state.blob });
    if (!resp.ok) {
      showStatus('ошибка');
      document.getElementById('text-meta').innerHTML =
        '<span class="err">' + (await resp.text()) + '</span>';
      return;
    }
    const data = await resp.json();
    state.boxes = data.lines.map(line => line.box);
    state.texts = data.lines.map(line => ({ text: line.text, confidence: line.confidence }));
    state.labels = state.texts.map(info => info.text || '');
    state.saved = state.boxes.map(() => false);
    state.selected = state.boxes.length ? 0 : -1;
    render();
    document.getElementById('text-out').textContent = data.text || '(пусто)';
    document.getElementById('text-meta').textContent =
      (data.auto_detected ? 'рамок найдено автоматически: ' : 'рамок задано вручную: ') +
      data.lines.length + ' | средняя уверенность: ' + (data.confidence * 100).toFixed(1) +
      '% | размер: ' + data.image_size.join('x');
    showStatus('готово');
  } catch (e) {
    showStatus('ошибка: ' + e);
  } finally {
    state.busy = false;
  }
}

// ---- загрузка файла --------------------------------------------------------
function loadFile(file) {
  state.blob = file;
  showStatus('загружаю...');
  photo.onload = () => {
    state.natW = photo.naturalWidth;
    state.natH = photo.naturalHeight;
    state.boxes = [];
    state.texts = [];
    state.selected = -1;
    editor.hidden = false;
    toolbar.hidden = false;
    document.getElementById('coords-tools').hidden = false;
    document.getElementById('coords').hidden = false;
    render();
    recognize(true);          // сразу показываем авто-разметку
  };
  photo.src = URL.createObjectURL(file);
}

function setAddMode(on) {
  state.addMode = on;
  document.getElementById('btn-add').classList.toggle('active', on);
  showStatus(on ? 'нарисуйте новую рамку на картинке' : 'готово');
}

drop.onclick = () => fileInput.click();
fileInput.onchange = () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); };
['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add('hover'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove('hover'); }));
drop.addEventListener('drop', ev => {
  const f = ev.dataTransfer.files[0]; if (f) loadFile(f); });

// ---- правка рамок: перетаскивание, ресайз, создание ------------------------
overlay.addEventListener('pointerdown', evt => {
  if (evt.button !== 0) return;
  const target = evt.target;
  const pos = toImage(evt);
  if (target.dataset && target.dataset.h) {
    const idx = Number(target.dataset.idx);
    drag = { mode: 'resize', idx, handle: target.dataset.h, start: pos };
  } else if (target.dataset && target.dataset.idx !== undefined) {
    const idx = Number(target.dataset.idx);
    state.selected = idx;
    drag = { mode: 'move', idx, start: pos, orig: state.boxes[idx].slice() };
    render();
  } else if (state.addMode) {
    const x = Math.max(0, Math.min(state.natW, pos[0]));
    const y = Math.max(0, Math.min(state.natH, pos[1]));
    state.boxes.push([x, y, x, y]);
    state.texts.push({ text: '', confidence: null });
    state.selected = state.boxes.length - 1;
    drag = { mode: 'create', idx: state.selected, start: [x, y] };
    setAddMode(false);
    render();
  } else {
    state.selected = -1;
    render();
    return;
  }
  if (drag) {
    if (drag.orig === undefined) drag.orig = state.boxes[drag.idx].slice();
    overlay.setPointerCapture(evt.pointerId);
    evt.preventDefault();
  }
});

overlay.addEventListener('pointermove', evt => {
  if (!drag) return;
  const pos = toImage(evt);
  const clampX = v => Math.max(0, Math.min(state.natW, v));
  const clampY = v => Math.max(0, Math.min(state.natH, v));
  const minSide = 4;
  if (drag.mode === 'move') {
    const [x0, y0, x1, y1] = drag.orig;
    const w = x1 - x0, h = y1 - y0;
    const nx = clampX(x0 + (pos[0] - drag.start[0]));
    const ny = clampY(y0 + (pos[1] - drag.start[1]));
    state.boxes[drag.idx] = [Math.min(nx, state.natW - w), Math.min(ny, state.natH - h),
                             Math.min(nx, state.natW - w) + w, Math.min(ny, state.natH - h) + h];
  } else if (drag.mode === 'resize') {
    let [x0, y0, x1, y1] = drag.orig;
    const px = clampX(pos[0]), py = clampY(pos[1]);
    if (drag.handle.includes('w')) x0 = Math.min(px, x1 - minSide);
    if (drag.handle.includes('e')) x1 = Math.max(px, x0 + minSide);
    if (drag.handle.includes('n')) y0 = Math.min(py, y1 - minSide);
    if (drag.handle.includes('s')) y1 = Math.max(py, y0 + minSide);
    state.boxes[drag.idx] = [x0, y0, x1, y1];
  } else if (drag.mode === 'create') {
    state.boxes[drag.idx] = [Math.min(drag.start[0], clampX(pos[0])),
                             Math.min(drag.start[1], clampY(pos[1])),
                             Math.max(drag.start[0], clampX(pos[0])),
                             Math.max(drag.start[1], clampY(pos[1]))];
  }
  // текст сбрасывается только если рамку действительно изменили
  const before = drag.orig, after = state.boxes[drag.idx];
  const moved = Math.abs(after[0] - before[0]) >= 1 || Math.abs(after[1] - before[1]) >= 1 ||
                Math.abs(after[2] - before[2]) >= 1 || Math.abs(after[3] - before[3]) >= 1;
  if (moved) state.texts[drag.idx] = { text: '', confidence: null };
  render();
  evt.preventDefault();
});

overlay.addEventListener('pointerup', evt => {
  if (!drag) return;
  const box = state.boxes[drag.idx];
  const tooSmall = Math.abs(box[2] - box[0]) < 6 || Math.abs(box[3] - box[1]) < 6;
  if (drag.mode === 'create' && tooSmall) {
    state.boxes.splice(drag.idx, 1);
    state.texts.splice(drag.idx, 1);
    state.selected = -1;
  }
  drag = null;
  render();
  evt.preventDefault();
});
overlay.addEventListener('pointercancel', () => { drag = null; });

// ---- список рамок и кнопки -------------------------------------------------
resultsBox.addEventListener('input', evt => {
  const field = evt.target.closest('input.label');
  if (field) state.labels[Number(field.dataset.idx)] = field.value;
});

resultsBox.addEventListener('click', async evt => {
  const button = evt.target.closest('button.save');
  if (button) {
    await saveLabel(Number(button.dataset.idx));
    return;
  }
  const coord = evt.target.closest('span.coord');
  if (coord) {
    const index = Number(coord.dataset.idx);
    state.selected = index;
    render();
    copyText((index + 1) + ': ' + state.boxes[index].map(v => Math.round(v)).join(', '),
             'координаты рамки ' + (index + 1));
    return;
  }
  if (evt.target.closest('input')) return;   // это ввод текста, а не выбор рамки
  const item = evt.target.closest('li');
  if (!item) return;
  state.selected = Number(item.dataset.idx);
  render();
  overlay.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
});

document.getElementById('btn-copy').onclick = () =>
  copyText(coordsLines().join('\n'), 'координаты');
document.getElementById('btn-json').onclick = () =>
  copyText(JSON.stringify(state.boxes.map(b => b.map(v => Math.round(v)))), 'JSON');
document.getElementById('btn-apply').onclick = () => applyCoords();

document.getElementById('btn-recognize').onclick = () => recognize(false);
document.getElementById('btn-auto').onclick = () => recognize(true);
document.getElementById('btn-add').onclick = () => setAddMode(!state.addMode);
document.getElementById('btn-del').onclick = () => {
  if (state.selected < 0) return showStatus('сначала выберите рамку');
  state.boxes.splice(state.selected, 1);
  state.texts.splice(state.selected, 1);
  state.selected = -1;
  render();
  recognize(false);
};
document.getElementById('btn-dup').onclick = () => {
  if (state.selected < 0) return showStatus('сначала выберите рамку');
  const [x0, y0, x1, y1] = state.boxes[state.selected];
  const shift = 8;
  state.boxes.splice(state.selected + 1, 0,
    [x0 + shift, y0 + shift, x1 + shift, y1 + shift]);
  state.texts.splice(state.selected + 1, 0, { text: '', confidence: null });
  state.selected += 1;
  render();
};
document.getElementById('btn-clear').onclick = () => {
  state.boxes = [];
  state.texts = [];
  state.selected = -1;
  document.getElementById('text-out').textContent = '';
  document.getElementById('text-meta').textContent = '';
  render();
  showStatus('рамки убраны');
};

document.addEventListener('keydown', evt => {
  if (evt.key === 'Escape') { setAddMode(false); return; }
  if ((evt.key === 'Delete' || evt.key === 'Backspace') && state.selected >= 0 &&
      !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    state.boxes.splice(state.selected, 1);
    state.texts.splice(state.selected, 1);
    state.selected = -1;
    render();
    evt.preventDefault();
  }
});

window.addEventListener('resize', () => { if (state.natW) render(); });


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
refreshLabelCount();
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
    lexicon_recognizer: ocr.Recognizer | None = None
    checkpoint: str | None = None
    label_dir: Path | None = None
    lock = threading.Lock()
    server_version = "hwocr/0.1"

    def _labels_dir(self) -> Path:
        """Folder with hand-labelled crops (used by ``train --real-dir``)."""
        path = self.label_dir or (PROJECT_ROOT / "data" / "real")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _labels_count(self) -> int:
        labels = self._labels_dir() / "labels.tsv"
        if not labels.exists():
            return 0
        return sum(1 for line in labels.read_text(encoding="utf-8").splitlines() if line.strip())

    def _picker(self, lexicon: bool) -> "ocr.Recognizer":
        """Recogniser with or without lexicon correction (cached)."""
        if not lexicon:
            return self.recognizer
        if self.lexicon_recognizer is None:
            self.lexicon_recognizer = ocr.Recognizer.load(
                self.checkpoint or DEFAULT_CHECKPOINT, lexicon_decode=True
            )
        return self.lexicon_recognizer

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
                "labels": self._labels_count(),
                "label_dir": str(self._labels_dir()),
            })
            return
        if path == "/api/labels":
            self._send_json({"count": self._labels_count(),
                             "dir": str(self._labels_dir())})
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path, _, query = self.path.partition("?")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            if path == "/api/recognize":
                self._recognize(body, segment="segment=0" not in query)
            elif path == "/api/label":
                self._save_label(body)
            elif path == "/api/demo":
                self._demo(json.loads(body.decode("utf-8") or "{}"))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            self._send(500, f"{type(exc).__name__}: {exc}".encode("utf-8"), "text/plain")

    # -- endpoints -------------------------------------------------------
    def _boxes_from_header(self) -> list[list[int]] | None:
        """Optional hand-edited line boxes: ``X-Boxes: [[x0,y0,x1,y1], ...]``."""
        raw = self.headers.get("X-Boxes")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        boxes: list[list[int]] = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                try:
                    boxes.append([int(round(float(value))) for value in item[:4]])
                except (TypeError, ValueError):
                    continue
        return boxes or None

    def _box_from_header(self, name: str = "X-Box") -> list[int] | None:
        """A single box from a header like ``X-Box: [x0,y0,x1,y1]``."""
        raw = self.headers.get(name)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list) or len(data) < 4:
            return None
        try:
            return [int(round(float(value))) for value in data[:4]]
        except (TypeError, ValueError):
            return None

    def _save_label(self, body: bytes) -> None:
        """Stores a hand-labelled crop (box + corrected text) for fine-tuning."""
        box = self._box_from_header("X-Box")
        raw_label = self.headers.get("X-Label")
        if box is None or not raw_label:
            self._send_json({"error": "X-Box and X-Label headers are required"}, code=400)
            return
        try:
            text = base64.b64decode(raw_label).decode("utf-8").strip()
        except Exception:
            self._send_json({"error": "X-Label must be base64 of the UTF-8 text"}, code=400)
            return
        if not text:
            self._send_json({"error": "empty label"}, code=400)
            return
        try:
            with Image.open(io.BytesIO(body)) as handle:
                image = np.asarray(ImageOps.exif_transpose(handle).convert("L"),
                                   dtype=np.uint8)
        except Exception:
            self._send_json({"error": "unsupported image format"}, code=400)
            return

        height, width = image.shape[:2]
        x0, y0, x1, y1 = box
        x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
        y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
        if x1 - x0 < 4 or y1 - y0 < 4:
            self._send_json({"error": "box is too small"}, code=400)
            return

        directory = self._labels_dir()
        count = self._labels_count()
        name = f"real_{count:05d}.png"
        Image.fromarray(image[y0:y1, x0:x1]).save(directory / name)
        with (directory / "labels.tsv").open("a", encoding="utf-8") as handle:
            handle.write(f"{name}\t{bidi.logical_to_visual(text)}\t{text}\n")
        self._send_json({"saved": name, "count": count + 1, "text": text,
                         "dir": str(directory)})

    def _recognize(self, body: bytes, segment: bool = True) -> None:
        if not body:
            self._send_json({"error": "empty request body"}, code=400)
            return
        try:
            # PIL (not cv2) so that the EXIF orientation of phone photos is applied
            with Image.open(io.BytesIO(body)) as handle:
                image = np.asarray(ImageOps.exif_transpose(handle).convert("L"),
                                   dtype=np.uint8)
        except Exception:
            image = None
        if image is None or image.size == 0:
            self._send_json({"error": "unsupported image format"}, code=400)
            return
        boxes = self._boxes_from_header()
        lexicon = self.headers.get("X-Lexicon") == "1"
        with self.lock:  # one inference at a time: the machine has no GPU
            page = self._picker(lexicon).recognize_page(image, segment=segment, boxes=boxes)
        payload = page.as_dict()
        payload["auto_detected"] = boxes is None
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
          port: int = 8009, label_dir: str | Path | None = None) -> None:
    """Load the checkpoint and serve the UI until interrupted.

    ``label_dir`` is where hand-labelled crops are written (``data/real`` by
    default); the training command picks them up via ``--real-dir``.
    """
    checkpoint = str(checkpoint)
    Handler.recognizer = ocr.Recognizer.load(checkpoint)
    Handler.checkpoint = checkpoint
    Handler.label_dir = Path(label_dir) if label_dir else None
    resolved_labels = Handler.label_dir or (PROJECT_ROOT / "data" / "real")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[web] hwocr UI on http://{host}:{port}  (checkpoint: {checkpoint})")
    print(f"[web] hand-labelled crops go to: {resolved_labels}")
    print("[web] Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[web] stopped")
    finally:
        server.server_close()


