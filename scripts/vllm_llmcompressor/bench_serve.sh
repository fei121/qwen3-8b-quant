#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL="${MODEL:?set MODEL to a Hugging Face id or local compressed model path}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
RUN_NAME="${RUN_NAME:?set RUN_NAME, for example baseline_bf16}"
DTYPE="${DTYPE:-auto}"
PORT="${PORT:-8000}"
OUT_BASE="${OUT_BASE:-${DATA_ROOT}/outputs}"
OUT_DIR="${OUT_BASE}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --trust-remote-code \
  > "${OUT_DIR}/serve.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

vllm bench serve \
  --backend vllm \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 128 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-dir "${OUT_DIR}" \
  --result-filename serve_bench.json
