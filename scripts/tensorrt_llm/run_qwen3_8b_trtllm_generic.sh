#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/qwen3_8b_tensorrt_llm}"
ENGINE_ROOT="${ENGINE_ROOT:-${OUTPUT_ROOT}/engines}"
TASKS="${TASKS:-gsm8k ceval-valid}"
RUNS="${RUNS:-bf16 int8_sq}"
CALIB_SIZE="${CALIB_SIZE:-128}"
CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-1024}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-4096}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-512}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
LIMIT="${LIMIT:-}"
GATHER_CONTEXT_LOGITS="${GATHER_CONTEXT_LOGITS:-1}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_step() {
  local name="$1"
  shift
  local log_dir="${OUTPUT_ROOT}/logs"
  local marker_dir="${log_dir}/markers"
  local done_file="${marker_dir}/${name}.done"
  local failed_file="${marker_dir}/${name}.failed"
  local log_file="${log_dir}/${name}.log"
  mkdir -p "${log_dir}" "${marker_dir}"
  if [[ -f "${done_file}" ]]; then
    echo "[$(timestamp)] SKIP ${name}"
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

precision_arg() {
  case "$1" in
    bf16) echo "bf16" ;;
    int8_sq) echo "int8_sq" ;;
    *) echo "unknown" ;;
  esac
}

mkdir -p "${OUTPUT_ROOT}" "${ENGINE_ROOT}"
echo "[$(timestamp)] Qwen3-8B TensorRT-LLM lm_eval workflow"
echo "MODEL_DIR=${MODEL_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "ENGINE_ROOT=${ENGINE_ROOT}"
echo "RUNS=${RUNS}"
echo "TASKS=${TASKS}"
echo "GATHER_CONTEXT_LOGITS=${GATHER_CONTEXT_LOGITS}"

for run_name in ${RUNS}; do
  precision="$(precision_arg "${run_name}")"
  if [[ "${precision}" == "unknown" ]]; then
    echo "unknown run: ${run_name}" >&2
    exit 2
  fi

  build_args=()
  if [[ "${GATHER_CONTEXT_LOGITS}" == "1" || "${GATHER_CONTEXT_LOGITS}" == "true" || "${GATHER_CONTEXT_LOGITS}" == "True" ]]; then
    build_args+=(--gather-context-logits)
  fi

  run_step "build_${run_name}_engine" \
    python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b.py \
      --model-dir "${MODEL_DIR}" \
      --output-root "${ENGINE_ROOT}" \
      --precision "${precision}" \
      --calib-size "${CALIB_SIZE}" \
      --calib-max-seq-length "${CALIB_MAX_SEQ_LENGTH}" \
      --build-engine \
      --delete-checkpoint-after-build \
      --max-batch-size "${MAX_BATCH_SIZE}" \
      --max-input-len "${MAX_INPUT_LEN}" \
      --max-seq-len "$((MAX_INPUT_LEN + MAX_OUTPUT_LEN))" \
      --max-num-tokens "${MAX_NUM_TOKENS}" \
      "${build_args[@]}"

  for task in ${TASKS}; do
    run_step "eval_${task}_${run_name}" \
      env \
        MODEL="${ENGINE_ROOT}/${run_name}/engine" \
        TOKENIZER="${MODEL_DIR}" \
        RUN_NAME="${run_name}" \
        TASK="${task}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}/eval" \
        DTYPE=bfloat16 \
        MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
        MAX_INPUT_LEN="${MAX_INPUT_LEN}" \
        MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        LIMIT="${LIMIT}" \
        ENABLE_THINKING=False \
        bash scripts/tensorrt_llm/eval_lmeval_trtllm.sh
  done
done

echo "[$(timestamp)] workflow finished"
