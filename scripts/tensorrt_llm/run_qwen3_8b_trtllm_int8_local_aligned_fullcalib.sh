#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
OUT_ROOT="${OUT_ROOT:-/path/to/outputs/qwen3_8b_trtllm_int8_local_aligned_fullcalib}"
GSM8K_CALIB="${GSM8K_CALIB:-${OUT_ROOT}/calib/gsm8k_train_chat_seed42_full}"
CEVAL_CALIB="${CEVAL_CALIB:-${OUT_ROOT}/calib/ceval_dev_chat_seed42_full}"
GSM8K_ENGINE_ROOT="${GSM8K_ENGINE_ROOT:-${OUT_ROOT}/engines/gsm8k_train_full}"
CEVAL_ENGINE_ROOT="${CEVAL_ENGINE_ROOT:-${OUT_ROOT}/engines/ceval_dev_full}"
GSM8K_CALIB_SIZE="${GSM8K_CALIB_SIZE:-7473}"
CEVAL_CALIB_SIZE="${CEVAL_CALIB_SIZE:-260}"

export MODEL_DIR OUT_ROOT GSM8K_CALIB CEVAL_CALIB
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export MPI4PY_MPIABI="${MPI4PY_MPIABI:-openmpi}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export PRTE_ALLOW_RUN_AS_ROOT=1
export PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1
export LD_LIBRARY_PATH="${CONDA_ROOT:-/path/to/miniconda3}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT_ROOT}"/{logs,calib,engines,eval,markers}
LOG_DIR="${OUT_ROOT}/logs"
MARKER_DIR="${OUT_ROOT}/markers"

timestamp() {
  date "+%F %T"
}

run_step() {
  local name="$1"
  shift
  local marker="${MARKER_DIR}/${name}.done"
  if [[ -f "${marker}" ]]; then
    echo "[$(timestamp)] SKIP ${name}"
    return 0
  fi
  echo "[$(timestamp)] START ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  touch "${marker}"
  echo "[$(timestamp)] DONE ${name}"
}

cleanup_old_outputs() {
  rm -rf \
    /path/to/outputs/qwen3_8b_trtllm_int8_fullcalib_per_dataset \
    /path/to/outputs/qwen3_8b_trtllm_int8_sq \
    /path/to/outputs/qwen3_8b_trtllm_int8_sq_2048
}

prepare_gsm8k_calib() {
  python - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

model_dir = os.environ["MODEL_DIR"]
output_dir = Path(os.environ["GSM8K_CALIB"])
seed = 42

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    use_fast=True,
)
dataset = load_dataset("gsm8k", "main", split="train")
dataset = dataset.shuffle(seed=seed)
rows = []
for item in dataset:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Solve the math problem step by step.",
        },
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": item["answer"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    rows.append({"text": text})

output_dir.mkdir(parents=True, exist_ok=True)
with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

metadata = {
    "source": "gsm8k/main",
    "split": "train",
    "seed": seed,
    "size": len(rows),
    "format": "local_aligned_chat_system_user_assistant",
}
(output_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(metadata, ensure_ascii=False))
PY
}

prepare_ceval_calib() {
  python scripts/tensorrt_llm/prepare_ceval_calib.py \
    --tokenizer "${MODEL_DIR}" \
    --output-dir "${CEVAL_CALIB}" \
    --split dev \
    --limit "${CEVAL_CALIB_SIZE}" \
    --seed 42
}

build_gsm8k_engine() {
  python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b.py \
    --model-dir "${MODEL_DIR}" \
    --output-root "${GSM8K_ENGINE_ROOT}" \
    --precision int8_sq \
    --dtype bfloat16 \
    --calib-dataset "${GSM8K_CALIB}" \
    --calib-size "${GSM8K_CALIB_SIZE}" \
    --batch-size 1 \
    --calib-max-seq-length 2048 \
    --tokenizer-max-seq-length 4096 \
    --build-engine \
    --delete-checkpoint-after-build \
    --max-batch-size 1 \
    --max-input-len 1792 \
    --max-seq-len 2112 \
    --max-num-tokens 1792
}

build_ceval_engine() {
  python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b.py \
    --model-dir "${MODEL_DIR}" \
    --output-root "${CEVAL_ENGINE_ROOT}" \
    --precision int8_sq \
    --dtype bfloat16 \
    --calib-dataset "${CEVAL_CALIB}" \
    --calib-size "${CEVAL_CALIB_SIZE}" \
    --batch-size 1 \
    --calib-max-seq-length 1024 \
    --tokenizer-max-seq-length 4096 \
    --build-engine \
    --delete-checkpoint-after-build \
    --max-batch-size 1 \
    --max-input-len 1280 \
    --max-seq-len 1536 \
    --max-num-tokens 1280 \
    --gather-context-logits
}

eval_gsm8k() {
  MODEL="${GSM8K_ENGINE_ROOT}/int8_sq/engine" \
  TOKENIZER="${MODEL_DIR}" \
  RUN_NAME="gsm8k_local_aligned_fullcalib_int8_sq" \
  TASK=gsm8k \
  OUTPUT_ROOT="${OUT_ROOT}/eval/gsm8k_train_full" \
  DTYPE=bfloat16 \
  MAX_BATCH_SIZE=1 \
  MAX_INPUT_LEN=1792 \
  MAX_OUTPUT_LEN=320 \
  BATCH_SIZE=1 \
  NUM_FEWSHOT=5 \
  ENABLE_THINKING=False \
  PROMPT_LOGPROBS=0 \
    bash scripts/tensorrt_llm/eval_lmeval_trtllm.sh
}

eval_ceval() {
  MODEL="${CEVAL_ENGINE_ROOT}/int8_sq/engine" \
  TOKENIZER="${MODEL_DIR}" \
  RUN_NAME="ceval_local_aligned_fullcalib_int8_sq" \
  TASK=ceval-valid \
  OUTPUT_ROOT="${OUT_ROOT}/eval/ceval_dev_full" \
  DTYPE=bfloat16 \
  MAX_BATCH_SIZE=1 \
  MAX_INPUT_LEN=1280 \
  MAX_OUTPUT_LEN=32 \
  BATCH_SIZE=1 \
  NUM_FEWSHOT=5 \
  ENABLE_THINKING=False \
  PROMPT_LOGPROBS=20 \
    bash scripts/tensorrt_llm/eval_lmeval_trtllm.sh
}

summarize_results() {
  python - <<'PY'
import glob
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
summary = {}
for path in glob.glob(str(out_root / "eval" / "**" / "*.json"), recursive=True):
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:
        summary[path] = {"error": str(exc)}
        continue
    metrics = {}
    for task, values in data.get("results", {}).items():
        metrics[task] = {
            key: value
            for key, value in values.items()
            if isinstance(value, (int, float)) or key in {"alias"}
        }
    summary[path] = metrics

(out_root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

echo "[$(timestamp)] OUT_ROOT=${OUT_ROOT}"
echo "[$(timestamp)] MODEL_DIR=${MODEL_DIR}"
df -h / "${DATA_ROOT:-.}" || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true

run_step cleanup_old_outputs cleanup_old_outputs
run_step prepare_gsm8k_calib prepare_gsm8k_calib
run_step build_gsm8k_engine build_gsm8k_engine
run_step eval_gsm8k eval_gsm8k
run_step prepare_ceval_calib prepare_ceval_calib
run_step build_ceval_engine build_ceval_engine
run_step eval_ceval eval_ceval
run_step summarize_results summarize_results

echo "[$(timestamp)] FINISHED"
