#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common/env.sh"

python -m pip install --upgrade pip
python -m pip install \
  "vllm>=0.14.0" \
  llmcompressor \
  transformers \
  accelerate \
  datasets \
  sentencepiece \
  protobuf \
  safetensors \
  numpy \
  tqdm \
  "lm-eval[api]>=0.4.9.2" \
  jsonlines \
  "chardet<6"
python -m pip freeze > "${PROJECT_ROOT}/requirements-lock.txt"
