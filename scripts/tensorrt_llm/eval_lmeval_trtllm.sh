#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL="${MODEL:?set MODEL to a TensorRT-LLM engine path, TRT checkpoint path, or HF model path}"
TOKENIZER="${TOKENIZER:-/path/to/Qwen3-8B}"
RUN_NAME="${RUN_NAME:-trtllm_bf16}"
TASK="${TASK:-gsm8k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs/qwen3_8b_trtllm_lmeval}"
LMEVAL_RUNTIME="${LMEVAL_RUNTIME:-/path/to/lm_eval_runtime}"
DTYPE="${DTYPE:-bfloat16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-4096}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
LIMIT="${LIMIT:-}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-20}"

mkdir -p "${OUTPUT_ROOT}/${RUN_NAME}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

export PYTHONPATH="${LMEVAL_RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false

model_args="model=${MODEL},tokenizer=${TOKENIZER},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE},max_batch_size=${MAX_BATCH_SIZE},max_input_len=${MAX_INPUT_LEN},max_output_len=${MAX_OUTPUT_LEN},enable_thinking=${ENABLE_THINKING},prompt_logprobs=${PROMPT_LOGPROBS}"

set +e
python -m lm_eval run \
  --model trtllm \
  --model_args "${model_args}" \
  --tasks "${TASK}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  "${limit_args[@]}" \
  --output_path "${OUTPUT_ROOT}/${RUN_NAME}/${TASK}.json" \
  --log_samples
exit_code=$?
set -e

if [[ "${exit_code}" -eq 139 ]] && find "${OUTPUT_ROOT}/${RUN_NAME}" -maxdepth 1 -name "*.json" -size +0c | grep -q .; then
  echo "WARNING: lm_eval exited with 139 after writing results; treating as success for TensorRT-LLM teardown segfault." >&2
  exit 0
fi

exit "${exit_code}"
