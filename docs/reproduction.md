# 复现指南

本仓库保留的是轻量、可公开的实验材料：脚本、汇总结果、图表和报告。模型权重、TensorRT engines、raw samples 和大日志不包含在仓库中。

## 轻量检查

先确认本地仓库结构、链接和测试都正常：

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python tools/check_release.py
```

`requirements-dev.txt` 只包含开发与检查所需依赖。真正的 GPU 量化和推理实验需要单独安装 vLLM、LLM Compressor、TensorRT-LLM、ModelOpt 或 AutoRound。

## GPU 工作流入口

所有 GPU 脚本都尽量通过环境变量传入路径。复现时请把下面的路径替换成你自己机器上的模型目录和输出目录。

### vLLM + LLM Compressor

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export OUTPUT_ROOT=/path/to/outputs/qwen3_8b_vllm
bash scripts/vllm_llmcompressor/run_qwen3_8b_gsm8k.sh
bash scripts/vllm_llmcompressor/run_qwen3_8b_ceval.sh
```

关键脚本：

- `scripts/vllm_llmcompressor/quant_int8_w8a8.py`：SmoothQuant + GPTQ 的 INT8 W8A8 导出。
- `scripts/vllm_llmcompressor/quant_mxfp4a16.py`：MXFP4A16 导出。
- `scripts/vllm_llmcompressor/report_gsm8k.py`：GSM8K 精度与 benchmark 汇总。
- `scripts/vllm_llmcompressor/report_ceval.py`：C-Eval 汇总。

### TensorRT-LLM + ModelOpt

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export OUTPUT_ROOT=/path/to/outputs/qwen3_8b_trtllm
export CALIB_DATASET=/path/to/calibration.jsonl
export CALIB_SIZE=260
bash scripts/tensorrt_llm/run_qwen3_8b_trtllm_generic.sh
```

根因消融入口：

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export BASE_OUT=/path/to/outputs/down_proj_ablation
export REFERENCE_OUT=/path/to/reference_int8_fullcalib
export DATASETS=ceval,gsm8k
export SKIP_MODULES=mlp.down_proj
bash scripts/tensorrt_llm/run_qwen3_8b_trtllm_int8_cumulative_down_o_ablation.sh
```

公开的累计消融汇总位于 [`../results/tensorrt_llm_down_proj_ablation.csv`](../results/tensorrt_llm_down_proj_ablation.csv)。

### AutoRound

INT8 W8A8：

```bash
export MODEL=/path/to/Qwen3-8B
export RUN_ROOT=/path/to/outputs/autoround_int8
bash scripts/autoround/int8_w8a8/run_autoround_quant.sh
bash scripts/autoround/int8_w8a8/run_vllm_eval.sh
```

MXFP4 fakequant：

```bash
export MODEL=/path/to/Qwen3-8B
export RUN_ROOT=/path/to/outputs/autoround_mxfp4_fakequant
bash scripts/autoround/mxfp4_fakequant/run_autoround_mxfp4_fakequant.sh
bash scripts/autoround/mxfp4_fakequant/run_vllm_eval.sh
```

## 复现时需要注意

- AutoRound MXFP4 fakequant 只作为精度参考，不作为真实性能 benchmark。
- TensorRT-LLM 与 vLLM 的速度数据来自不同 harness，不能直接当作同一 benchmark 横向比较。
- 不同 GPU、驱动、CUDA、batch 设置和 serving 参数都会影响吞吐。建议把本仓库结果作为相对参考，而不是逐 token/s 的绝对目标。
- C-Eval 是多选 log-likelihood 排序任务，GSM8K 是生成式数学推理任务；两者对量化误差的敏感点不同，曲线不一定单调一致。
