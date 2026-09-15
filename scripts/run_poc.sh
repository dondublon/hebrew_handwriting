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
LETTER_CHECKPOINT=${LETTER_CHECKPOINT:-models/letters_pretrained.pt}
STEPS_SHORT=${STEPS_SHORT:-400}
STEPS_MEDIUM=${STEPS_MEDIUM:-500}
STEPS_LIGHT=${STEPS_LIGHT:-1500}
STEPS_HEAVY=${STEPS_HEAVY:-1000}

echo "== 1/6 self-test =============================================="
$PY -m hwocr.cli selftest --quick

echo "== 2/6 font registry =========================================="
$PY -m hwocr.cli list-fonts

echo "== 3/6 export synthetic datasets =============================="
$PY -m hwocr.cli generate --pango 200

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "== 4/6 warm start on isolated letters ========================="
  $PY -m hwocr.cli pretrain --out "$LETTER_CHECKPOINT"

  echo "== 5/6 train the full recogniser (CTC) ========================"
  $PY -m hwocr.cli train --checkpoint "$CHECKPOINT" --init-from "$LETTER_CHECKPOINT" \
      --steps-short "$STEPS_SHORT" --steps-medium "$STEPS_MEDIUM" \
      --steps-light "$STEPS_LIGHT" --steps-heavy "$STEPS_HEAVY"
fi

echo "== 6/6 evaluation ============================================="
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
