#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-8B}"
INT8_DIR="${INT8_DIR:-${DATA_ROOT}/models/qwen3_8b_int8_w8a8}"
MXFP4_DIR="${MXFP4_DIR:-${DATA_ROOT}/models/qwen3_8b_mxfp4a16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs_qwen3_8b}"
CEVAL_TASK="${CEVAL_TASK:-ceval-valid}"
CEVAL_NUM_FEWSHOT="${CEVAL_NUM_FEWSHOT:-5}"
CEVAL_MAX_MODEL_LEN="${CEVAL_MAX_MODEL_LEN:-4096}"
CEVAL_GPU_MEMORY_UTILIZATION="${CEVAL_GPU_MEMORY_UTILIZATION:-0.85}"
LOG_ROOT="${OUTPUT_ROOT}/ceval_run_logs"
MARKER_ROOT="${LOG_ROOT}/markers"
mkdir -p "${LOG_ROOT}" "${MARKER_ROOT}" "${OUTPUT_ROOT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_step() {
  local name="$1"
  shift
  local done_file="${MARKER_ROOT}/${name}.done"
  local failed_file="${MARKER_ROOT}/${name}.failed"
  local log_file="${LOG_ROOT}/${name}.log"
  if [[ -f "${done_file}" ]]; then
    echo "[$(timestamp)] SKIP ${name}: ${done_file} exists"
    return 0
  fi
  rm -f "${failed_file}"
  echo "[$(timestamp)] START ${name}"
  echo "[$(timestamp)] COMMAND: $*" > "${log_file}"
  if "$@" >> "${log_file}" 2>&1; then
    echo "[$(timestamp)] DONE ${name}" | tee -a "${log_file}"
    date -Iseconds > "${done_file}"
  else
    local exit_code=$?
    echo "[$(timestamp)] FAILED ${name} exit=${exit_code}" | tee -a "${log_file}"
    echo "${exit_code}" > "${failed_file}"
    return "${exit_code}"
  fi
}

eval_ceval() {
  local model="$1"
  local dtype="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" lm_eval \
    --model vllm \
    --model_args pretrained="${model}",dtype="${dtype}",add_bos_token=true,gpu_memory_utilization="${CEVAL_GPU_MEMORY_UTILIZATION}",max_model_len="${CEVAL_MAX_MODEL_LEN}",trust_remote_code=True \
    --tasks "${CEVAL_TASK}" \
    --num_fewshot "${CEVAL_NUM_FEWSHOT}" \
    --batch_size auto \
    --output_path "${out_dir}/ceval.json"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[$(timestamp)] Qwen3-8B C-Eval workflow started"
echo "MODEL_DIR=${MODEL_DIR}"
echo "INT8_DIR=${INT8_DIR}"
echo "MXFP4_DIR=${MXFP4_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "CEVAL_TASK=${CEVAL_TASK}"
echo "CEVAL_NUM_FEWSHOT=${CEVAL_NUM_FEWSHOT}"

run_step 00_check_env python scripts/common/check_env.py

run_step 01_ceval_baseline_bf16 \
  eval_ceval "${MODEL_DIR}" auto "${OUTPUT_ROOT}/baseline_bf16"

run_step 02_ceval_int8_w8a8 \
  eval_ceval "${INT8_DIR}" auto "${OUTPUT_ROOT}/int8_w8a8"

run_step 03_ceval_mxfp4a16 \
  eval_ceval "${MXFP4_DIR}" auto "${OUTPUT_ROOT}/mxfp4a16"

run_step 04_ceval_summary \
  python scripts/vllm_llmcompressor/report_ceval.py \
    --output-root "${OUTPUT_ROOT}" \
    --model-label "Qwen/Qwen3-8B" \
    --task-name "${CEVAL_TASK}"

echo "[$(timestamp)] Qwen3-8B C-Eval workflow finished"
