#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
INPUT=${INPUT:-/orange/pinaki.sarder/kirill.luka/mmai/text_fe/nephrology_consults.csv}
OUTPUT=${OUTPUT:-$ROOT/outputs/nephrology_structured.jsonl}
MODEL=${MODEL:-google/medgemma-27b-text-it}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
WORKERS=${WORKERS:-8}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m text_extraction.structured \
  --input "$INPUT" \
  --input-format csv \
  --text-field NOTE_TEXT \
  --id-field note_deiden_id \
  --output "$OUTPUT" \
  --schema "$ROOT/schemas/nephrology_renal.json" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --workers "$WORKERS" \
  "$@"
