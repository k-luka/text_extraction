#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
INPUT=${INPUT:-/orange/pinaki.sarder/singletarya/mmai/text_fe/discharge_summaries.csv}
OUTPUT=${OUTPUT:-$ROOT/outputs/discharge_embeddings.npz}
MODEL=${MODEL:-emilyalsentzer/Bio_ClinicalBERT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-8}

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m text_extraction.embeddings \
  --input "$INPUT" \
  --input-format csv \
  --text-field NOTE_TEXT \
  --id-field note_deiden_id \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  "$@"

