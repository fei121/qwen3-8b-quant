#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/path/to/outputs/qwen3_8b_autoround_mxfp4_fakequant}"
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
LOG_DIR="$RUN_ROOT/logs"
MARKER_DIR="$RUN_ROOT/logs/markers"

mkdir -p "$LOG_DIR" "$MARKER_DIR" "$RUN_ROOT/results" "$RUN_ROOT/models"

run_step() {
  local name="$1"
  shift
  local done="$MARKER_DIR/$name.done"
  local failed="$MARKER_DIR/$name.failed"
  local log="$LOG_DIR/$name.log"
  if [[ -f "$done" ]]; then
    echo "[$(date -Iseconds)] SKIP $name"
    return 0
  fi
  rm -f "$failed"
  echo "[$(date -Iseconds)] START $name"
  echo "[$(date -Iseconds)] COMMAND: $*" > "$log"
  if "$@" >> "$log" 2>&1; then
    echo "[$(date -Iseconds)] DONE $name" | tee -a "$log"
    date -Iseconds > "$done"
  else
    rc=$?
    echo "[$(date -Iseconds)] FAILED $name rc=$rc" | tee -a "$log"
    echo "$rc" > "$failed"
    return "$rc"
  fi
}

run_step 01_quant_gsm8k_autoround_mxfp4_fake \
  env RUN_ROOT="$RUN_ROOT" MODEL_DIR="$MODEL_DIR" CALIB_NAME=gsm8k \
    CALIB_DATASET=/path/to/calib/gsm8k_train_chat_seed42_512.jsonl \
    OUT_DIR=/path/to/models/qwen3_8b_autoround_mxfp4_gsm8kcalib_fake \
    NSAMPLES=384 SEQLEN=128 BATCH_SIZE=1 ITERS=0 \
    "$RUN_ROOT/workflow/run_autoround_mxfp4_fakequant.sh"

run_step 02_eval_gsm8k_autoround_mxfp4_fake \
  env RUN_ROOT="$RUN_ROOT" \
    MODEL=/path/to/models/qwen3_8b_autoround_mxfp4_gsm8kcalib_fake \
    TASK=gsm8k RUN_NAME=gsm8k_autoround_mxfp4_gsm8kcalib_fake \
    NUM_FEWSHOT=5 MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.85 \
    "$RUN_ROOT/workflow/run_vllm_eval.sh"

run_step 03_quant_ceval_autoround_mxfp4_fake \
  env RUN_ROOT="$RUN_ROOT" MODEL_DIR="$MODEL_DIR" CALIB_NAME=ceval \
    CALIB_DATASET=/path/to/calib/ceval_dev_chat_seed42.jsonl \
    OUT_DIR=/path/to/models/qwen3_8b_autoround_mxfp4_cevalcalib_fake \
    NSAMPLES=255 SEQLEN=128 BATCH_SIZE=1 ITERS=0 \
    "$RUN_ROOT/workflow/run_autoround_mxfp4_fakequant.sh"

run_step 04_eval_ceval_autoround_mxfp4_fake \
  env RUN_ROOT="$RUN_ROOT" \
    MODEL=/path/to/models/qwen3_8b_autoround_mxfp4_cevalcalib_fake \
    TASK=ceval-valid RUN_NAME=ceval_autoround_mxfp4_cevalcalib_fake \
    NUM_FEWSHOT=5 MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.85 \
    "$RUN_ROOT/workflow/run_vllm_eval.sh"

echo "[$(date -Iseconds)] all dataset-serial AutoRound MXFP4 fakequant jobs finished"
