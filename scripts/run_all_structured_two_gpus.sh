#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-google/medgemma-27b-text-it}
PYTHON_BIN=${PYTHON_BIN:-python}
WORKERS=${WORKERS:-8}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-300}
LOG_DIR=${LOG_DIR:-$ROOT/outputs/logs}
DISCHARGE_OUTPUT=${DISCHARGE_OUTPUT:-$ROOT/outputs/discharge_structured.jsonl}
NEPHROLOGY_OUTPUT=${NEPHROLOGY_OUTPUT:-$ROOT/outputs/nephrology_structured.jsonl}

mkdir -p "$LOG_DIR"

server_pids=()
extraction_pids=()

cleanup() {
  for pid in "${extraction_pids[@]}" "${server_pids[@]}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_server() {
  local gpu=$1
  local port=$2
  GPU="$gpu" PORT="$port" MODEL="$MODEL" PYTHON_BIN="$PYTHON_BIN" \
    "$ROOT/scripts/serve_medgemma.sh" >"$LOG_DIR/medgemma_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
}

wait_until_ready() {
  local port=$1
  local pid=$2
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM on port $port exited during startup; see $LOG_DIR" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM on port $port; see $LOG_DIR" >&2
      return 1
    fi
    sleep 2
  done
}

start_server 0 8000
start_server 1 8001
wait_until_ready 8000 "${server_pids[0]}"
wait_until_ready 8001 "${server_pids[1]}"

MODEL="$MODEL" PYTHON_BIN="$PYTHON_BIN" WORKERS="$WORKERS" \
  OUTPUT="$DISCHARGE_OUTPUT" \
  BASE_URL=http://127.0.0.1:8000/v1 \
  "$ROOT/scripts/run_discharge_structured.sh" "$@" &
extraction_pids+=("$!")

MODEL="$MODEL" PYTHON_BIN="$PYTHON_BIN" WORKERS="$WORKERS" \
  OUTPUT="$NEPHROLOGY_OUTPUT" \
  BASE_URL=http://127.0.0.1:8001/v1 \
  "$ROOT/scripts/run_nephrology_structured.sh" "$@" &
extraction_pids+=("$!")

status=0
for pid in "${extraction_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
