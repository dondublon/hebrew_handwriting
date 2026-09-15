#!/usr/bin/env bash
# Full proof-of-concept pipeline: generate data -> train -> evaluate -> demo.
#
#   bash scripts/run_poc.sh            # default: ~15-20 min on a 6-core CPU
#   SKIP_TRAIN=1 bash scripts/run_poc.sh
#
# Everything is written to data/, models/ and runs/ (all git-ignored).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-python3}
CHECKPOINT=${CHECKPOINT:-models/crnn_gveret.pt}
STEPS_PLAIN=${STEPS_PLAIN:-250}
STEPS_LIGHT=${STEPS_LIGHT:-400}
STEPS_HEAVY=${STEPS_HEAVY:-300}

echo "== 1/5 self-test =============================================="
$PY -m hwocr.cli selftest --quick

echo "== 2/5 font registry =========================================="
$PY -m hwocr.cli list-fonts

echo "== 3/5 export synthetic datasets =============================="
$PY -m hwocr.cli generate

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "== 4/5 train ================================================"
  $PY -m hwocr.cli train --checkpoint "$CHECKPOINT" \
      --steps-plain "$STEPS_PLAIN" --steps-light "$STEPS_LIGHT" --steps-heavy "$STEPS_HEAVY"
fi

echo "== 5/5 evaluation ============================================="
$PY -m hwocr.cli eval --checkpoint "$CHECKPOINT" --save-errors 20 \
    --errors-dir runs/errors

echo "-- Tesseract baseline (same data) ----------------------------"
$PY -m hwocr.cli baseline --split data/test_clean --limit 100 --checkpoint "$CHECKPOINT" || true

echo "-- demo: render a phrase with the font and recognise it ------"
$PY -m hwocr.cli demo --checkpoint "$CHECKPOINT" --lines 3 --out runs/demo.png || true
$PY -m hwocr.cli demo --checkpoint "$CHECKPOINT" --heavy 1.0 --out runs/demo_photo.png || true

echo
echo "Done. Start the web UI with:"
echo "  $PY -m hwocr.cli serve --checkpoint $CHECKPOINT"
