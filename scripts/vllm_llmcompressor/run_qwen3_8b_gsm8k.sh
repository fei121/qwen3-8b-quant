#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-8B}"
INT8_DIR="${INT8_DIR:-${DATA_ROOT}/models/qwen3_8b_int8_w8a8}"
MXFP4_DIR="${MXFP4_DIR:-${DATA_ROOT}/models/qwen3_8b_mxfp4a16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/outputs_qwen3_8b}"
LOG_ROOT="${OUTPUT_ROOT}/full_run_logs"
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

eval_gsm8k() {
  local model="$1"
  local dtype="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" lm_eval \
    --model vllm \
    --model_args pretrained="${model}",dtype="${dtype}",add_bos_token=true,gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path "${out_dir}/gsm8k.json"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-512}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-2048}"
export SEED="${SEED:-42}"

echo "[$(timestamp)] Qwen3-8B experiment started"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

run_step 00_check_env python scripts/common/check_env.py

run_step 01_gsm8k_baseline_bf16 \
  eval_gsm8k "${MODEL_DIR}" auto "${OUTPUT_ROOT}/baseline_bf16"

run_step 02_gsm8k_baseline_fp16 \
  eval_gsm8k "${MODEL_DIR}" float16 "${OUTPUT_ROOT}/baseline_fp16"

run_step 03_quant_int8_w8a8 \
  env MODEL_ID="${MODEL_DIR}" OUT_DIR="${INT8_DIR}" python scripts/vllm_llmcompressor/quant_int8_w8a8.py

run_step 04_gsm8k_int8_w8a8 \
  eval_gsm8k "${INT8_DIR}" auto "${OUTPUT_ROOT}/int8_w8a8"

if [[ -f "${MXFP4_DIR}/config.json" && -f "${MXFP4_DIR}/model.safetensors" ]]; then
  echo "[$(timestamp)] SKIP 05_quant_mxfp4a16: existing model found"
  date -Iseconds > "${MARKER_ROOT}/05_quant_mxfp4a16.done"
fi

run_step 05_quant_mxfp4a16 \
  env MODEL_ID="${MODEL_DIR}" OUT_DIR="${MXFP4_DIR}" python scripts/vllm_llmcompressor/quant_mxfp4a16.py

run_step 06_gsm8k_mxfp4a16 \
  eval_gsm8k "${MXFP4_DIR}" auto "${OUTPUT_ROOT}/mxfp4a16"

run_step 07_offline_bench_baseline_bf16 \
  python scripts/vllm_llmcompressor/bench_offline.py \
    --model "${MODEL_DIR}" \
    --run-name baseline_bf16 \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --output-dir "${OUTPUT_ROOT}"

run_step 08_offline_bench_int8_w8a8 \
  python scripts/vllm_llmcompressor/bench_offline.py \
    --model "${INT8_DIR}" \
    --run-name int8_w8a8 \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --output-dir "${OUTPUT_ROOT}"

run_step 09_offline_bench_mxfp4a16 \
  python scripts/vllm_llmcompressor/bench_offline.py \
    --model "${MXFP4_DIR}" \
    --run-name mxfp4a16 \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --output-dir "${OUTPUT_ROOT}"

run_step 10_serve_bench_baseline_bf16 \
  env RUN_NAME=baseline_bf16 MODEL="${MODEL_DIR}" DTYPE=auto PORT=8010 OUT_BASE="${OUTPUT_ROOT}" \
    bash scripts/vllm_llmcompressor/bench_serve.sh

run_step 11_serve_bench_int8_w8a8 \
  env RUN_NAME=int8_w8a8 MODEL="${INT8_DIR}" DTYPE=auto PORT=8011 OUT_BASE="${OUTPUT_ROOT}" \
    bash scripts/vllm_llmcompressor/bench_serve.sh

run_step 12_serve_bench_mxfp4a16 \
  env RUN_NAME=mxfp4a16 MODEL="${MXFP4_DIR}" DTYPE=auto PORT=8012 OUT_BASE="${OUTPUT_ROOT}" \
    bash scripts/vllm_llmcompressor/bench_serve.sh

run_step 13_collect_report_visuals \
  python scripts/vllm_llmcompressor/report_gsm8k.py \
    --output-root "${OUTPUT_ROOT}" \
    --model-label "Qwen/Qwen3-8B"

echo "[$(timestamp)] Qwen3-8B experiment finished"
