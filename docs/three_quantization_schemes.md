# Qwen3-8B Three-Scheme Quantization Comparison

This report compares local Qwen3-8B quantization results across three implementation ecosystems:

1. **Intel AutoRound INT8 W8A8**, exported by AutoRound and evaluated with vLLM.
2. **Intel AutoRound MXFP4 fakequant**, used as an accuracy-only low-bit reference.
3. **NVIDIA TensorRT-LLM INT8 SmoothQuant**, exported through ModelOpt and evaluated as TensorRT-LLM engines.

Two BF16 references are included:

- vLLM BF16 baseline.
- TensorRT-LLM BF16 engine baseline.

All GSM8K and C-Eval results are 5-shot. GSM8K reports `exact_match,flexible-extract`; C-Eval valid reports `acc`.

## Accuracy Overview

| Ecosystem | Method | Precision | GSM8K flexible | C-Eval acc | Notes |
|---|---|---:|---:|---:|---|
| vLLM | BF16 baseline | BF16 | 0.8802 | 0.7905 | vLLM full-precision reference |
| TensorRT-LLM | BF16 baseline | BF16 | 0.8848 | 0.7853 | TensorRT-LLM engine reference |
| AutoRound | AutoRound | INT8 W8A8 | 0.8749 | 0.7764 | Accuracy stays close to vLLM BF16 |
| AutoRound | AutoRound | MXFP4 fakequant | 0.8613 | 0.7667 | Fakequant; not a real speed benchmark |
| TensorRT-LLM | ModelOpt SmoothQuant | INT8 W8A8 | 0.7983 | 0.6872 | Large accuracy drop in the tested configuration |

Additional vLLM ecosystem references:

| Ecosystem | Method | Precision | GSM8K flexible | C-Eval acc |
|---|---|---:|---:|---:|
| vLLM | LLM Compressor SmoothQuant + GPTQ | INT8 W8A8 | 0.8719 | 0.7853 |
| vLLM | LLM Compressor MXFP4A16 | MXFP4A16 | 0.8643 | 0.7608 |

## Accuracy Delta Against BF16

| Method | Baseline | GSM8K delta | C-Eval delta |
|---|---|---:|---:|
| AutoRound INT8 W8A8 | vLLM BF16 | -0.0053 | -0.0141 |
| AutoRound MXFP4 fakequant | vLLM BF16 | -0.0190 | -0.0238 |
| LLM Compressor INT8 W8A8 | vLLM BF16 | -0.0083 | -0.0052 |
| LLM Compressor MXFP4A16 | vLLM BF16 | -0.0159 | -0.0297 |
| TensorRT-LLM INT8 SmoothQuant | TensorRT-LLM BF16 | -0.0864 | -0.0981 |

AutoRound INT8 and LLM Compressor INT8 both preserve accuracy well. TensorRT-LLM INT8 SmoothQuant gives speedup in its own workflow but loses much more accuracy before additional tuning.

## Speed Notes

The collected speed data uses two different measurement styles:

- vLLM benchmark throughput from offline and serve benchmarks.
- End-to-end evaluation wall time from lm-eval/TensorRT-LLM wrappers.

These numbers should not be mixed as if they were the same benchmark. The reliable TensorRT-LLM statement is intra-workflow: INT8 SmoothQuant was faster than the TensorRT-LLM BF16 engine for both GSM8K and C-Eval in this setup.

## Conclusion

- **Best INT8 accuracy retention**: AutoRound INT8 and vLLM LLM Compressor INT8.
- **Best low-bit accuracy reference**: AutoRound MXFP4 fakequant and vLLM MXFP4A16 are close enough to motivate further low-bit experiments, but only vLLM MXFP4A16 is a real deployment-style benchmark here.
- **Most important anomaly**: TensorRT-LLM INT8 SmoothQuant dropped far more accuracy than expected, motivating the root-cause study in [`tensorrt_llm_int8_down_proj_analysis.md`](tensorrt_llm_int8_down_proj_analysis.md).

Structured data lives in [`../results/qwen3_8b_quantization_summary.csv`](../results/qwen3_8b_quantization_summary.csv).
