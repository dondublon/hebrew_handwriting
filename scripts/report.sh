#!/usr/bin/env bash
# Reproducible evaluation report: numbers, baseline comparison and demos.
#
#   bash scripts/report.sh                 # uses models/crnn_gveret.pt
#   CHECKPOINT=... LIMIT=100 bash scripts/report.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-python3}
CHECKPOINT=${CHECKPOINT:-models/crnn_gveret.pt}
LIMIT=${LIMIT:-60}

echo "== metrics on every exported split (raw + lexicon-corrected) =="
$PY -m hwocr.cli eval --checkpoint "$CHECKPOINT" \
    --save-errors 15 --errors-dir runs/errors --lexicon-decode

echo
echo "== Tesseract baseline on the same images ====================="
$PY -m hwocr.cli baseline --split data/test_clean --limit "$LIMIT" \
    --checkpoint "$CHECKPOINT"

echo
echo "== demos ====================================================="
$PY -m hwocr.cli demo --checkpoint "$CHECKPOINT" --lines 3 --out runs/demo.png || true
$PY -m hwocr.cli demo --checkpoint "$CHECKPOINT" --heavy 1.0 --out runs/demo_photo.png || true

if [ -f runs/page_demo.png ]; then
  echo
  echo "== page level recognition ===================================="
  $PY -m hwocr.cli recognize --image runs/page_demo.png \
      --out-image runs/annotated.png --checkpoint "$CHECKPOINT"
fi

echo
echo "Done. Error examples: runs/errors/"
