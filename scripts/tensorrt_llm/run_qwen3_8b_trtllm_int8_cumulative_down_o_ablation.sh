#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
source "${CONDA_ROOT:-/path/to/miniconda3}/etc/profile.d/conda.sh"
conda activate base
source "${PROJECT_ROOT}/scripts/common/env.sh"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
BASE_OUT="${BASE_OUT:-/path/to/outputs/qwen3_8b_trtllm_int8_cumulative_down_o_ablation}"
REFERENCE_OUT="${REFERENCE_OUT:-/path/to/outputs/qwen3_8b_trtllm_int8_local_aligned_fullcalib}"
GSM8K_CALIB="${GSM8K_CALIB:-${REFERENCE_OUT}/calib/gsm8k_train_chat_seed42_full}"
CEVAL_CALIB="${CEVAL_CALIB:-${REFERENCE_OUT}/calib/ceval_dev_chat_seed42_full}"
GSM8K_CALIB_SIZE="${GSM8K_CALIB_SIZE:-7473}"
CEVAL_CALIB_SIZE="${CEVAL_CALIB_SIZE:-260}"
NUM_LAYERS="${NUM_LAYERS:-36}"
DATASETS="${DATASETS:-ceval,gsm8k}"
SKIP_MODULES="${SKIP_MODULES:-mlp.down_proj,self_attn.o_proj}"

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
  local dataset="$1"
  local label="$2"
  local first_n="$3"
  local layer_end=$((first_n - 1))
  local skip_layers="0-${layer_end}"
  local work_dir="${BASE_OUT}/work/${dataset}_${label}"
  cleanup_work_dir "${work_dir}"

  common_args=(
    --model-dir "${MODEL_DIR}"
    --output-root "${work_dir}"
    --skip-modules "${SKIP_MODULES}"
    --skip-layers "${skip_layers}"
    --dtype bfloat16
    --batch-size 1
    --tokenizer-max-seq-length 4096
    --delete-checkpoint-after-build
    --max-batch-size 1
  )

  if [[ "${dataset}" == "ceval" ]]; then
    python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b_skip_modules.py \
      "${common_args[@]}" \
      --calib-dataset "${CEVAL_CALIB}" \
      --calib-size "${CEVAL_CALIB_SIZE}" \
      --calib-max-seq-length 1024 \
      --max-input-len 1280 \
      --max-seq-len 1536 \
      --max-num-tokens 1280 \
      --gather-context-logits
  elif [[ "${dataset}" == "gsm8k" ]]; then
    python scripts/tensorrt_llm/export_modelopt_trtllm_qwen3_8b_skip_modules.py \
      "${common_args[@]}" \
      --calib-dataset "${GSM8K_CALIB}" \
      --calib-size "${GSM8K_CALIB_SIZE}" \
      --calib-max-seq-length 2048 \
      --max-input-len 1792 \
      --max-seq-len 2112 \
      --max-num-tokens 1792
  else
    echo "unknown dataset: ${dataset}" >&2
    return 2
  fi
}

eval_dataset() {
  local dataset="$1"
  local label="$2"
  local work_dir="${BASE_OUT}/work/${dataset}_${label}"

  if [[ "${dataset}" == "ceval" ]]; then
    MODEL="${work_dir}/int8_sq/engine" \
    TOKENIZER="${MODEL_DIR}" \
    RUN_NAME="${label}_ceval_int8_sq" \
    TASK=ceval-valid \
    OUTPUT_ROOT="${BASE_OUT}/eval/ceval/${label}" \
    DTYPE=bfloat16 \
    MAX_BATCH_SIZE=1 \
    MAX_INPUT_LEN=1280 \
    MAX_OUTPUT_LEN=32 \
    BATCH_SIZE=1 \
    NUM_FEWSHOT=5 \
    ENABLE_THINKING=False \
    PROMPT_LOGPROBS=20 \
      bash scripts/tensorrt_llm/eval_lmeval_trtllm.sh
  elif [[ "${dataset}" == "gsm8k" ]]; then
    MODEL="${work_dir}/int8_sq/engine" \
    TOKENIZER="${MODEL_DIR}" \
    RUN_NAME="${label}_gsm8k_int8_sq" \
    TASK=gsm8k \
    OUTPUT_ROOT="${BASE_OUT}/eval/gsm8k/${label}" \
    DTYPE=bfloat16 \
    MAX_BATCH_SIZE=1 \
    MAX_INPUT_LEN=1792 \
    MAX_OUTPUT_LEN=320 \
    BATCH_SIZE=1 \
    NUM_FEWSHOT=5 \
    ENABLE_THINKING=False \
    PROMPT_LOGPROBS=0 \
      bash scripts/tensorrt_llm/eval_lmeval_trtllm.sh
  else
    echo "unknown dataset: ${dataset}" >&2
    return 2
  fi
}

archive_metadata_and_delete_model() {
  local dataset="$1"
  local label="$2"
  local work_dir="${BASE_OUT}/work/${dataset}_${label}"
  local artifact_dir="${BASE_OUT}/artifacts/${dataset}/${label}"
  mkdir -p "${artifact_dir}"
  if [[ -f "${work_dir}/int8_sq/metadata.json" ]]; then
    cp -f "${work_dir}/int8_sq/metadata.json" "${artifact_dir}/metadata.json"
  fi
  find "${BASE_OUT}/eval/${dataset}/${label}" -type f -name "*.json" -print > "${artifact_dir}/eval_json_files.txt" 2>/dev/null || true
  cleanup_work_dir "${work_dir}"
}

summarize_results() {
  python - <<'PY'
import csv
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

def add_result(dataset, name, first_n, path):
    data = read_json(path)
    if not data:
        return
    for task, values in data.get("results", {}).items():
        if dataset == "ceval" and task != "ceval-valid":
            continue
        if dataset == "gsm8k" and task != "gsm8k":
            continue
        rows.append({
            "dataset": dataset,
            "name": name,
            "first_n_layers": first_n,
            "task": task,
            "acc": values.get("acc,none"),
            "acc_norm": values.get("acc_norm,none"),
            "exact_match_strict": values.get("exact_match,strict-match"),
            "exact_match_flexible": values.get("exact_match,flexible-extract"),
            "path": str(path),
        })

if reference:
    for path in glob.glob(str(reference / "eval" / "ceval_dev_full" / "**" / "*.json"), recursive=True):
        add_result("ceval", "baseline_int8_sq_fullcalib", 0, path)
    for path in glob.glob(str(reference / "eval" / "gsm8k_train_full" / "**" / "*.json"), recursive=True):
        add_result("gsm8k", "baseline_int8_sq_fullcalib", 0, path)

for dataset in ("ceval", "gsm8k"):
    for path in glob.glob(str(base / "eval" / dataset / "*" / "**" / "*.json"), recursive=True):
        parts = Path(path).relative_to(base).parts
        name = parts[2] if len(parts) > 2 else Path(path).parent.name
        first_n = None
        if name.startswith("first_") and "_layers" in name:
            try:
                first_n = int(name.split("_")[1])
            except Exception:
                first_n = None
        add_result(dataset, name, first_n, path)

summary = {
    "experiment": "Qwen3-8B TensorRT-LLM ModelOpt INT8 SmoothQuant cumulative first-N-layer down/o skip ablation",
    "calibration": {
        "ceval": {"dataset": os.environ["CEVAL_CALIB"], "size": int(os.environ["CEVAL_CALIB_SIZE"])},
        "gsm8k": {"dataset": os.environ["GSM8K_CALIB"], "size": int(os.environ["GSM8K_CALIB_SIZE"])},
    },
    "skip_modules": os.environ["SKIP_MODULES"],
    "num_layers": int(os.environ["NUM_LAYERS"]),
    "rows": rows,
}
(base / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

with (base / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["dataset", "name", "first_n_layers", "task", "acc", "acc_norm", "exact_match_strict", "exact_match_flexible", "path"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Qwen3-8B TensorRT-LLM INT8 Cumulative Down/O Skip Ablation",
    "",
    f"- Skip modules: `{os.environ['SKIP_MODULES']}`",
    f"- First-N layer range: `1..{os.environ['NUM_LAYERS']}` using internal layers `0..N-1`",
    f"- C-Eval calibration: `{os.environ['CEVAL_CALIB']}` (`{os.environ['CEVAL_CALIB_SIZE']}` samples)",
    f"- GSM8K calibration: `{os.environ['GSM8K_CALIB']}` (`{os.environ['GSM8K_CALIB_SIZE']}` samples)",
    "",
    "| dataset | run | first N | metric | value | result |",
    "|---|---|---:|---|---:|---|",
]
for row in rows:
    if row["dataset"] == "ceval":
        metric, value = "acc", row["acc"]
    else:
        metric, value = "exact_match_flexible", row["exact_match_flexible"]
    if value is None:
        continue
    lines.append(f"| {row['dataset']} | {row['name']} | {row['first_n_layers']} | {metric} | {value:.6f} | `{row['path']}` |")
(base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

run_experiment() {
  local dataset="$1"
  local first_n="$2"
  local label
  label="$(printf 'first_%02d_layers_down_o' "${first_n}")"
  run_step "build_${dataset}_${label}" build_engine "${dataset}" "${label}" "${first_n}"
  run_step "eval_${dataset}_${label}" eval_dataset "${dataset}" "${label}"
  run_step "archive_cleanup_${dataset}_${label}" archive_metadata_and_delete_model "${dataset}" "${label}"
  run_step "summarize_after_${dataset}_${label}" summarize_results
}

export BASE_OUT REFERENCE_OUT GSM8K_CALIB CEVAL_CALIB GSM8K_CALIB_SIZE CEVAL_CALIB_SIZE NUM_LAYERS SKIP_MODULES

echo "[$(timestamp)] output root: ${BASE_OUT}"
echo "[$(timestamp)] model dir: ${MODEL_DIR}"
echo "[$(timestamp)] datasets: ${DATASETS}"
echo "[$(timestamp)] skip modules: ${SKIP_MODULES}"
echo "[$(timestamp)] num layers: ${NUM_LAYERS}"

IFS=',' read -r -a dataset_list <<< "${DATASETS}"
for dataset in "${dataset_list[@]}"; do
  dataset="${dataset// /}"
  for first_n in $(seq 1 "${NUM_LAYERS}"); do
    run_experiment "${dataset}" "${first_n}"
  done
done

echo "[$(timestamp)] all experiments complete"
summarize_results
