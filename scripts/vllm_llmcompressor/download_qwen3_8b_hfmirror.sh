#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy LOCAL_VPN_PROXY

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
MAX_WORKERS="${MAX_WORKERS:-8}"

"${PYTHON_BIN}" - <<'PY'
import os

from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
max_workers = int(os.environ["MAX_WORKERS"])

path = snapshot_download(
    repo_id=repo_id,
    endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    local_dir=local_dir,
    resume_download=True,
    max_workers=max_workers,
)
print(path)
PY
