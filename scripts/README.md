# Scripts

The release keeps Qwen3-8B-only scripts.

| Directory | Purpose |
|---|---|
| `common/` | Shared JSON loading, score extraction, and metric normalization helpers. |
| `vllm_llmcompressor/` | vLLM + LLM Compressor quantization, evaluation, benchmarking, and reporting. |
| `tensorrt_llm/` | ModelOpt export, TensorRT-LLM build/eval workflows, activation diagnostics, and ablations. |
| `autoround/` | AutoRound INT8 W8A8 and MXFP4 fakequant workflows promoted from archived experiment scripts. |

Heavy runtime dependencies such as TensorRT-LLM, ModelOpt, vLLM, LLM Compressor, and AutoRound are intentionally not included in `requirements-dev.txt`.
