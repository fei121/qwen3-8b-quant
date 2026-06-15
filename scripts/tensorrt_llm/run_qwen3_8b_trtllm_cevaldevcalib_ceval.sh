#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/qwen3_8b_trtllm_int8_cevaldevcalib_ceval_$(date +%Y%m%d)}"
ENGINE_ROOT="${ENGINE_ROOT:-${OUTPUT_ROOT}/engines}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/eval}"
CALIB_DATASET="${CALIB_DATASET:-${PROJECT_ROOT}/calib_ceval_dev_chat_seed42}"
CALIB_SIZE="${CALIB_SIZE:-260}"
CALIB_LIMIT="${CALIB_LIMIT:-512}"
CALIB_SPLIT="${CALIB_SPLIT:-dev}"
CALIB_SEED="${CALIB_SEED:-42}"
CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-1024}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-512}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASK="${TASK:-ceval-valid}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
RUN_NAME="${RUN_NAME:-int8_sq_cevaldevcalib_ceval_full}"

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

mkdir -p "${OUTPUT_ROOT}/logs" "${ENGINE_ROOT}" "${EVAL_ROOT}" "${CALIB_DATASET}"

cat > "${OUTPUT_ROOT}/logs/params.log" <<EOF
run=int8_sq_ceval_dev_calib_ceval_1024_engine
calib_source=ceval/ceval-exam ${CALIB_SPLIT} shuffled seed=${CALIB_SEED} chat_template enable_thinking=${ENABLE_THINKING}
calib_dataset=${CALIB_DATASET}
calib_size=${CALIB_SIZE}
calib_limit=${CALIB_LIMIT}
calib_max_seq_length=${CALIB_MAX_SEQ_LENGTH}
task=${TASK}
num_fewshot=${NUM_FEWSHOT}
max_batch_size=${MAX_BATCH_SIZE}
batch_size=${BATCH_SIZE}
max_input_len=${MAX_INPUT_LEN}
max_output_len=${MAX_OUTPUT_LEN}
max_seq_len=$((MAX_INPUT_LEN + MAX_OUTPUT_LEN))
max_num_tokens=${MAX_NUM_TOKENS}
start=$(date -Iseconds)
EOF

run_step "prepare_ceval_dev_calib" \
  python scripts/tensorrt_llm/prepare_ceval_calib.py \
    --tokenizer "${MODEL_DIR}" \
    --output-dir "${CALIB_DATASET}" \
    --split "${CALIB_SPLIT}" \
    --seed "${CALIB_SEED}" \
    --limit "${CALIB_LIMIT}"

run_step "build_int8_sq" \
  python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b.py \
    --model-dir "${MODEL_DIR}" \
    --output-root "${ENGINE_ROOT}" \
    --precision int8_sq \
    --calib-dataset "${CALIB_DATASET}" \
    --calib-size "${CALIB_SIZE}" \
    --calib-max-seq-length "${CALIB_MAX_SEQ_LENGTH}" \
    --build-engine \
    --delete-checkpoint-after-build \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --max-input-len "${MAX_INPUT_LEN}" \
    --max-seq-len "$((MAX_INPUT_LEN + MAX_OUTPUT_LEN))" \
    --max-num-tokens "${MAX_NUM_TOKENS}" \
    --gather-context-logits

run_step "eval_${TASK}_${RUN_NAME}" \
  env \
    MODEL="${ENGINE_ROOT}/int8_sq/engine" \
    TOKENIZER="${MODEL_DIR}" \
    RUN_NAME="${RUN_NAME}" \
    TASK="${TASK}" \
    OUTPUT_ROOT="${EVAL_ROOT}" \
    DTYPE=bfloat16 \
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
    MAX_INPUT_LEN="${MAX_INPUT_LEN}" \
    MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_FEWSHOT="${NUM_FEWSHOT}" \
    ENABLE_THINKING="${ENABLE_THINKING}" \
    PROMPT_LOGPROBS=20 \
    bash scripts/tensorrt_llm/eval_lmeval_trtllm.sh

echo "[$(timestamp)] workflow finished: ${OUTPUT_ROOT}"
