#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
OUT_ROOT="${OUT_ROOT:-/path/to/outputs/qwen3_8b_trtllm_int8_fullcalib_per_dataset}"
GSM8K_CALIB="${GSM8K_CALIB:-${OUT_ROOT}/calib/gsm8k_train_full_chat_seed42}"
CEVAL_CALIB="${CEVAL_CALIB:-${OUT_ROOT}/calib/ceval_dev_full_chat_seed42}"
GSM8K_ENGINE_ROOT="${GSM8K_ENGINE_ROOT:-${OUT_ROOT}/engines/gsm8k_train_full}"
CEVAL_ENGINE_ROOT="${CEVAL_ENGINE_ROOT:-${OUT_ROOT}/engines/ceval_dev_full}"
GSM8K_CALIB_SIZE="${GSM8K_CALIB_SIZE:-7473}"
CEVAL_CALIB_SIZE="${CEVAL_CALIB_SIZE:-260}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-2048}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2560}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

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

cleanup_previous_temporary_outputs() {
  rm -rf \
    /path/to/outputs/qwen3_8b_trtllm_int8_sq \
    /path/to/outputs/qwen3_8b_trtllm_int8_sq_2048
}

prepare_gsm8k_calib() {
  python - <<'PY'
import json
import os
import random
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
rows = []
for item in dataset:
    content = f"Question: {item['question']}\nAnswer: {item['answer']}"
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Solve the problem step by step and give the final answer.",
        },
        {"role": "user", "content": content},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    rows.append({"text": text})

random.Random(seed).shuffle(rows)
output_dir.mkdir(parents=True, exist_ok=True)
with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

metadata = {
    "source": "gsm8k/main",
    "split": "train",
    "seed": seed,
    "size": len(rows),
    "format": "chat_template_question_answer",
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

build_engine() {
  local output_root="$1"
  local calib_dataset="$2"
  local calib_size="$3"
  python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b.py \
    --model-dir "${MODEL_DIR}" \
    --output-root "${output_root}" \
    --precision int8_sq \
    --dtype bfloat16 \
    --calib-dataset "${calib_dataset}" \
    --calib-size "${calib_size}" \
    --batch-size 1 \
    --calib-max-seq-length "${MAX_INPUT_LEN}" \
    --tokenizer-max-seq-length 4096 \
    --build-engine \
    --delete-checkpoint-after-build \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --max-input-len "${MAX_INPUT_LEN}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --max-num-tokens "${MAX_NUM_TOKENS}" \
    --gather-context-logits
}

eval_task() {
  local engine="$1"
  local task="$2"
  local run_name="$3"
  local eval_root="$4"
  MODEL="${engine}" \
  TOKENIZER="${MODEL_DIR}" \
  RUN_NAME="${run_name}" \
  TASK="${task}" \
  OUTPUT_ROOT="${eval_root}" \
  MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
  MAX_INPUT_LEN="${MAX_INPUT_LEN}" \
  MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN}" \
  BATCH_SIZE="${EVAL_BATCH_SIZE}" \
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

summary_path = out_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

echo "[$(timestamp)] OUT_ROOT=${OUT_ROOT}"
echo "[$(timestamp)] MODEL_DIR=${MODEL_DIR}"
df -h / "${DATA_ROOT:-.}" || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true

run_step cleanup_previous_temporary_outputs cleanup_previous_temporary_outputs
run_step prepare_gsm8k_calib prepare_gsm8k_calib
run_step build_gsm8k_engine build_engine "${GSM8K_ENGINE_ROOT}" "${GSM8K_CALIB}" "${GSM8K_CALIB_SIZE}"
run_step eval_gsm8k eval_task "${GSM8K_ENGINE_ROOT}/int8_sq/engine" gsm8k gsm8k_train_full_int8_sq "${OUT_ROOT}/eval/gsm8k_train_full"
run_step prepare_ceval_calib prepare_ceval_calib
run_step build_ceval_engine build_engine "${CEVAL_ENGINE_ROOT}" "${CEVAL_CALIB}" "${CEVAL_CALIB_SIZE}"
run_step eval_ceval eval_task "${CEVAL_ENGINE_ROOT}/int8_sq/engine" ceval-valid ceval_dev_full_int8_sq "${OUT_ROOT}/eval/ceval_dev_full"
run_step summarize_results summarize_results

echo "[$(timestamp)] FINISHED"
