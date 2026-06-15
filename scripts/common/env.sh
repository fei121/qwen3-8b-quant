#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "${DATA_ROOT}/models" \
  "${DATA_ROOT}/outputs" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}"

# AutoDL exposes this script to accelerate Hugging Face/model downloads.
if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

# Keep an explicitly forwarded local VPN proxy ahead of network_turbo.
if [[ -n "${LOCAL_VPN_PROXY:-}" ]]; then
  export HTTP_PROXY="${LOCAL_VPN_PROXY}"
  export HTTPS_PROXY="${LOCAL_VPN_PROXY}"
  export ALL_PROXY="${LOCAL_VPN_PROXY}"
  export http_proxy="${LOCAL_VPN_PROXY}"
  export https_proxy="${LOCAL_VPN_PROXY}"
  export all_proxy="${LOCAL_VPN_PROXY}"
fi
