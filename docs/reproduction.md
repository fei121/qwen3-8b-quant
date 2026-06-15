# Reproduction Guide

This repository keeps the lightweight, public pieces of the experiment: scripts, summary results, charts, and reports. It does not include model weights, TensorRT engines, raw samples, or large logs.

## Lightweight Checks

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python tools/check_release.py
```

## GPU Workflow Entry Points

The GPU scripts are environment-variable driven. Set local paths for your machine instead of relying on any private experiment directory.

### vLLM + LLM Compressor

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export OUTPUT_ROOT=/path/to/outputs/qwen3_8b_vllm
bash scripts/vllm_llmcompressor/run_qwen3_8b_gsm8k.sh
bash scripts/vllm_llmcompressor/run_qwen3_8b_ceval.sh
```

Important scripts:

- `scripts/vllm_llmcompressor/quant_int8_w8a8.py`
- `scripts/vllm_llmcompressor/quant_mxfp4a16.py`
- `scripts/vllm_llmcompressor/report_gsm8k.py`
- `scripts/vllm_llmcompressor/report_ceval.py`

### TensorRT-LLM + ModelOpt

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export OUTPUT_ROOT=/path/to/outputs/qwen3_8b_trtllm
export CALIB_DATASET=/path/to/calibration.jsonl
export CALIB_SIZE=260
bash scripts/tensorrt_llm/run_qwen3_8b_trtllm_generic.sh
```

For root-cause ablations:

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export BASE_OUT=/path/to/outputs/down_proj_ablation
export REFERENCE_OUT=/path/to/reference_int8_fullcalib
export DATASETS=ceval,gsm8k
export SKIP_MODULES=mlp.down_proj
bash scripts/tensorrt_llm/run_qwen3_8b_trtllm_int8_cumulative_down_o_ablation.sh
```

The public cumulative ablation summary is in `results/tensorrt_llm_down_proj_ablation.csv`.

### AutoRound

```bash
export MODEL=/path/to/Qwen3-8B
export RUN_ROOT=/path/to/outputs/autoround_int8
bash scripts/autoround/int8_w8a8/run_autoround_quant.sh
bash scripts/autoround/int8_w8a8/run_vllm_eval.sh
```

MXFP4 fakequant:

```bash
export MODEL=/path/to/Qwen3-8B
export RUN_ROOT=/path/to/outputs/autoround_mxfp4_fakequant
bash scripts/autoround/mxfp4_fakequant/run_autoround_mxfp4_fakequant.sh
bash scripts/autoround/mxfp4_fakequant/run_vllm_eval.sh
```

## Notes

- AutoRound MXFP4 fakequant is used as an accuracy reference, not as a real deployment-speed benchmark.
- TensorRT-LLM speed and vLLM speed are not directly comparable in this release because their benchmark harnesses are different.
- If you reproduce on a different GPU, use the structured results as relative reference points rather than exact expected values.
