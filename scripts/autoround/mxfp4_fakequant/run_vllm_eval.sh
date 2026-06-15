#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/path/to/outputs/qwen3_8b_autoround_mxfp4_fakequant}"
ENV="${ENV:-/path/to/envs/vllm-lc}"
MODEL="${MODEL:?MODEL required}"
TASK="${TASK:?TASK required}"
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
OUT_DIR="$RUN_ROOT/results/$RUN_NAME"
TMPDIR="${TMPDIR:-/path/to/tmp}"

mkdir -p "$OUT_DIR" "$RUN_ROOT/logs" "$TMPDIR"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

export TMPDIR
export HF_HOME="${HF_HOME:-/path/to/cache/huggingface_eval_autoround_mxfp4}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

if [[ ! -f "$MODEL/config.json" ]]; then
  mapfile -t model_children < <(find "$MODEL" -mindepth 1 -maxdepth 1 -type d -exec test -f "{}/config.json" \; -print 2>/dev/null | sort)
  if [[ "${#model_children[@]}" -eq 1 ]]; then
    MODEL="${model_children[0]}"
  fi
fi
echo "Using MODEL=$MODEL"

"$ENV/bin/python" -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "datasets==3.6.0"

"$ENV/bin/lm_eval" \
  --model vllm \
  --model_args pretrained="$MODEL",dtype=auto,add_bos_token=true,gpu_memory_utilization="$GPU_MEMORY_UTILIZATION",max_model_len="$MAX_MODEL_LEN" \
  --tasks "$TASK" \
  --num_fewshot "$NUM_FEWSHOT" \
  --batch_size auto \
  --output_path "$OUT_DIR/${TASK}.json"
