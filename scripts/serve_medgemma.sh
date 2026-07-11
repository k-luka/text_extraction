#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
PORT=${PORT:-8000}
MODEL=${MODEL:-google/medgemma-27b-text-it}
PYTHON_BIN=${PYTHON_BIN:-python}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}

PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES="$GPU"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --generation-config vllm \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
