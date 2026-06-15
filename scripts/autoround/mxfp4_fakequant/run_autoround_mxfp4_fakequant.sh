#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/path/to/outputs/qwen3_8b_autoround_mxfp4_fakequant}"
ENV="${ENV:-/path/to/envs/vllm-lc}"
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
CALIB_NAME="${CALIB_NAME:?CALIB_NAME required}"
CALIB_DATASET="${CALIB_DATASET:?CALIB_DATASET required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
NSAMPLES="${NSAMPLES:-512}"
SEQLEN="${SEQLEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ITERS="${ITERS:-0}"
TMPDIR="${TMPDIR:-/path/to/tmp}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models" "$TMPDIR" "$(dirname "$OUT_DIR")"

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

export TMPDIR
export HF_HOME="${HF_HOME:-/path/to/cache/huggingface_autoround_mxfp4}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

"$ENV/bin/python" -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "datasets==4.0.0"

"$ENV/bin/python" - <<PY
import json
from pathlib import Path
from transformers import AutoTokenizer

model = "$MODEL_DIR"
path = Path("$CALIB_DATASET")
assert path.exists(), path
rows = []
with path.open("r", encoding="utf-8") as f:
    for i, line in zip(range(3), f):
        obj = json.loads(line)
        assert isinstance(obj.get("text"), str) and obj["text"].strip()
        rows.append(obj["text"])
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print({"calib_name": "$CALIB_NAME", "sample_count_checked": len(rows), "first_tokens": len(tok(rows[0], add_special_tokens=False).input_ids)})
PY

"$ENV/bin/auto-round" \
  --model "$MODEL_DIR" \
  --model_dtype bfloat16 \
  --dataset "$CALIB_DATASET" \
  --output_dir "$OUT_DIR" \
  --format llm_compressor \
  --scheme MXFP4 \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
  --batch_size "$BATCH_SIZE" \
  --iters "$ITERS" \
  --device 0 \
  --low_gpu_mem_usage

"$ENV/bin/python" - <<PY
import json
from pathlib import Path
p = Path("$OUT_DIR")
children = [x for x in p.iterdir() if x.is_dir()] if p.exists() else []
model_dirs = [x for x in children if (x / "config.json").exists()]

def fix_quantization_config(qc):
    if not isinstance(qc, dict):
        return
    for group in qc.get("config_groups", {}).values():
        weights = group.get("weights")
        if isinstance(weights, dict):
            weights.pop("is_mx", None)
            if weights.get("type") == "float" and weights.get("num_bits") == 4 and weights.get("group_size") == 32:
                weights["strategy"] = "group"
        group["input_activations"] = None
        group["format"] = None
    qc["ignore"] = ["lm_head"]
    qc.pop("provider", None)

def strip_is_mx(obj):
    if isinstance(obj, dict):
        obj.pop("is_mx", None)
        for value in obj.values():
            strip_is_mx(value)
    elif isinstance(obj, list):
        for value in obj:
            strip_is_mx(value)

for model_dir in model_dirs:
    for name in ("config.json", "quantization_config.json"):
        cfg_path = model_dir / name
        if not cfg_path.exists():
            continue
        data = json.loads(cfg_path.read_text())
        if name == "config.json":
            fix_quantization_config(data.get("quantization_config"))
        else:
            fix_quantization_config(data)
        strip_is_mx(data)
        cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print("vllm_mxfp4a16_compat", cfg_path)

print("OUT_DIR", p)
print("direct_config", (p / "config.json").exists())
print("direct_index", (p / "model.safetensors.index.json").exists())
print("direct_safetensors", len(list(p.glob("*.safetensors"))))
print("child_model_dirs", [str(x) for x in model_dirs])
for m in model_dirs:
    print("child", m, "safetensors", len(list(m.glob("*.safetensors"))), "index", (m / "model.safetensors.index.json").exists())
PY
