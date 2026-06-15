#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${CONDA_ROOT:-/path/to/miniconda3}/etc/profile.d/conda.sh"
conda activate base
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
BASE_OUT="${BASE_OUT:-/path/to/outputs/qwen3_8b_trtllm_int8_skip_layer_ablation}"
REFERENCE_OUT="${REFERENCE_OUT:-/path/to/outputs/qwen3_8b_trtllm_int8_local_aligned_fullcalib}"
CEVAL_CALIB="${CEVAL_CALIB:-${REFERENCE_OUT}/calib/ceval_dev_chat_seed42_full}"
CEVAL_CALIB_SIZE="${CEVAL_CALIB_SIZE:-260}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export MPI4PY_MPIABI="${MPI4PY_MPIABI:-openmpi}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export PRTE_ALLOW_RUN_AS_ROOT=1
export PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1
export LD_LIBRARY_PATH="${CONDA_ROOT:-/path/to/miniconda3}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${BASE_OUT}"/{artifacts,eval,logs,markers,work}
LOG_DIR="${BASE_OUT}/logs"
MARKER_DIR="${BASE_OUT}/markers"

timestamp() {
  date "+%F %T"
}

log_disk() {
  echo "[$(timestamp)] disk usage:"
  df -h "${DATA_ROOT:-.}" /
}

run_step() {
  local name="$1"
  shift
  local done_marker="${MARKER_DIR}/${name}.done"
  local failed_marker="${MARKER_DIR}/${name}.failed"
  if [[ -f "${done_marker}" ]]; then
    echo "[$(timestamp)] SKIP ${name}"
    return 0
  fi
  rm -f "${failed_marker}"
  echo "[$(timestamp)] START ${name}"
  log_disk
  set +e
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  local exit_code="${PIPESTATUS[0]}"
  set -e
  if [[ "${exit_code}" -ne 0 ]]; then
    touch "${failed_marker}"
    echo "[$(timestamp)] FAILED ${name} exit=${exit_code}" >&2
    exit "${exit_code}"
  fi
  touch "${done_marker}"
  echo "[$(timestamp)] DONE ${name}"
  log_disk
}

cleanup_work_dir() {
  local work_dir="$1"
  if [[ -n "${work_dir}" && "${work_dir}" == "${BASE_OUT}/work/"* ]]; then
    rm -rf "${work_dir}"
  fi
}

build_engine() {
  local label="$1"
  local skip_layers="$2"
  local work_dir="${BASE_OUT}/work/${label}"
  cleanup_work_dir "${work_dir}"
  python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b_skip_layers.py \
    --model-dir "${MODEL_DIR}" \
    --output-root "${work_dir}" \
    --calib-dataset "${CEVAL_CALIB}" \
    --calib-size "${CEVAL_CALIB_SIZE}" \
    --skip-layers "${skip_layers}" \
    --dtype bfloat16 \
    --batch-size 1 \
    --calib-max-seq-length 1024 \
    --tokenizer-max-seq-length 4096 \
    --delete-checkpoint-after-build \
    --max-batch-size 1 \
    --max-input-len 1280 \
    --max-seq-len 1536 \
    --max-num-tokens 1280 \
    --gather-context-logits
}

eval_ceval() {
  local label="$1"
  local work_dir="${BASE_OUT}/work/${label}"
  MODEL="${work_dir}/int8_sq/engine" \
  TOKENIZER="${MODEL_DIR}" \
  RUN_NAME="${label}_ceval_int8_sq" \
  TASK=ceval-valid \
  OUTPUT_ROOT="${BASE_OUT}/eval/${label}" \
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

archive_metadata_and_delete_model() {
  local label="$1"
  local work_dir="${BASE_OUT}/work/${label}"
  local artifact_dir="${BASE_OUT}/artifacts/${label}"
  mkdir -p "${artifact_dir}"
  if [[ -f "${work_dir}/int8_sq/metadata.json" ]]; then
    cp -f "${work_dir}/int8_sq/metadata.json" "${artifact_dir}/metadata.json"
  fi
  if [[ -d "${BASE_OUT}/eval/${label}" ]]; then
    find "${BASE_OUT}/eval/${label}" -type f -name "*.json" -print > "${artifact_dir}/eval_json_files.txt"
  fi
  cleanup_work_dir "${work_dir}"
}

summarize_results() {
  python - <<'PY'
import glob
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_OUT"])
reference = Path(os.environ.get("REFERENCE_OUT", ""))
rows = []

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

def add_result(name, path):
    data = read_json(path)
    if not data:
        return
    for task, values in data.get("results", {}).items():
        rows.append({
            "name": name,
            "task": task,
            "acc": values.get("acc,none"),
            "acc_norm": values.get("acc_norm,none"),
            "path": str(path),
        })

if reference:
    for path in glob.glob(str(reference / "eval" / "ceval_dev_full" / "**" / "*.json"), recursive=True):
        add_result("baseline_int8_sq_fullcalib", path)

for path in glob.glob(str(base / "eval" / "*" / "**" / "*.json"), recursive=True):
    parts = Path(path).relative_to(base).parts
    label = parts[1] if len(parts) > 1 else Path(path).parent.name
    add_result(label, path)

summary = {
    "experiment": "Qwen3-8B TensorRT-LLM ModelOpt INT8 SmoothQuant selective BF16 layer ablation",
    "calibration": {
        "dataset": os.environ["CEVAL_CALIB"],
        "size": int(os.environ["CEVAL_CALIB_SIZE"]),
    },
    "rows": rows,
}
(base / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# Qwen3-8B TensorRT-LLM INT8 Skip-Layer Ablation",
    "",
    f"- Calibration: `{os.environ['CEVAL_CALIB']}`",
    f"- Calibration size: `{os.environ['CEVAL_CALIB_SIZE']}`",
    "- Evaluation: `ceval-valid`, 5-shot, thinking disabled, prompt logprobs enabled",
    "- Cleanup: generated TensorRT-LLM checkpoint/engine directories are deleted after each group",
    "",
    "| run | task | acc | acc_norm | result |",
    "|---|---:|---:|---:|---|",
]
for row in rows:
    acc = "" if row["acc"] is None else f"{row['acc']:.6f}"
    acc_norm = "" if row["acc_norm"] is None else f"{row['acc_norm']:.6f}"
    lines.append(f"| {row['name']} | {row['task']} | {acc} | {acc_norm} | `{row['path']}` |")
(base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

run_experiment() {
  local label="$1"
  local skip_layers="$2"
  run_step "build_${label}" build_engine "${label}" "${skip_layers}"
  run_step "eval_${label}_ceval" eval_ceval "${label}"
  run_step "archive_cleanup_${label}" archive_metadata_and_delete_model "${label}"
  run_step "summarize_after_${label}" summarize_results
}

export BASE_OUT REFERENCE_OUT CEVAL_CALIB CEVAL_CALIB_SIZE

echo "[$(timestamp)] output root: ${BASE_OUT}"
echo "[$(timestamp)] model dir: ${MODEL_DIR}"
echo "[$(timestamp)] ceval calib: ${CEVAL_CALIB}"

run_experiment "skip_l16" "16"
run_experiment "skip_l16_l18" "16-18"
run_experiment "skip_l16_l21" "16-21"

echo "[$(timestamp)] all experiments complete"
summarize_results
