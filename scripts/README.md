# 脚本目录

本目录只保留 Qwen3-8B 公开复现实验需要的脚本。大依赖、模型权重、engine 产物和运行日志不随仓库发布。

| 目录 | 用途 |
|---|---|
| `common/` | JSON 读取、分数抽取、benchmark 指标归一化等通用工具。 |
| `vllm_llmcompressor/` | vLLM + LLM Compressor 的量化、评测、benchmark 与报告生成。 |
| `tensorrt_llm/` | ModelOpt 导出、TensorRT-LLM build/eval、激活诊断与消融实验。 |
| `autoround/` | 从归档实验中整理出的 AutoRound INT8 W8A8 与 MXFP4 fakequant 工作流。 |

`requirements-dev.txt` 只包含轻量开发依赖。TensorRT-LLM、ModelOpt、vLLM、LLM Compressor 和 AutoRound 需要根据目标机器与 CUDA / PyTorch 版本单独安装。
