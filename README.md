# Qwen3-8B Quantization Study

This repository is a reproducible, public release of a Qwen3-8B quantization study across three ecosystems:

- **vLLM + LLM Compressor**: BF16, INT8 W8A8, and MXFP4A16.
- **Intel AutoRound**: INT8 W8A8 and MXFP4 fakequant, evaluated with vLLM.
- **NVIDIA TensorRT-LLM + ModelOpt**: BF16 and INT8 SmoothQuant engines.

The short version: INT8 can preserve Qwen3-8B accuracy well, but the TensorRT-LLM / ModelOpt INT8 SmoothQuant configuration used here dropped much more accuracy than the vLLM and AutoRound flows. Follow-up ablations point to **`mlp.down_proj` W8A8 activation/weight quantization** as the dominant source of the TensorRT-LLM INT8 loss.

![Transformer MLP residual mechanism](assets/figures/mlp_residual_mechanism.png)

## Key Results

All reported GSM8K and C-Eval runs use 5-shot evaluation.

| Ecosystem | Method | Precision | GSM8K flexible | C-Eval acc | Main takeaway |
|---|---|---:|---:|---:|---|
| vLLM | BF16 baseline | BF16 | 0.8802 | 0.7905 | Full-precision reference |
| TensorRT-LLM | BF16 baseline | BF16 | 0.8848 | 0.7853 | Engine reference |
| AutoRound | AutoRound | INT8 W8A8 | 0.8749 | 0.7764 | Best accuracy retention among tested exported INT8 flows |
| AutoRound | AutoRound | MXFP4 fakequant | 0.8613 | 0.7667 | Accuracy-only low-bit reference |
| vLLM | LLM Compressor | INT8 W8A8 | 0.8719 | 0.7853 | Stable vLLM INT8 result |
| vLLM | LLM Compressor | MXFP4A16 | 0.8643 | 0.7608 | Real vLLM MXFP4A16 reference |
| TensorRT-LLM | ModelOpt SmoothQuant | INT8 W8A8 | 0.7983 | 0.6872 | Large accuracy drop before ablation |

Structured data:

- [`results/qwen3_8b_quantization_summary.csv`](results/qwen3_8b_quantization_summary.csv)
- [`results/qwen3_8b_quantization_summary.json`](results/qwen3_8b_quantization_summary.json)
- [`results/tensorrt_llm_down_proj_ablation.csv`](results/tensorrt_llm_down_proj_ablation.csv)
- [`results/tensorrt_llm_down_proj_ablation.json`](results/tensorrt_llm_down_proj_ablation.json)

## vLLM Inference Benchmark

The vLLM + LLM Compressor runs are the cleanest serving reference in this study. They use the same vLLM evaluation stack for BF16, INT8 W8A8, and MXFP4A16, so the accuracy and throughput trade-off is easier to compare than cross-backend wall-clock time.

![vLLM GSM8K accuracy](assets/figures/vllm_gsm8k_accuracy.svg)

Accuracy stayed close to the BF16 baseline:

| vLLM run | GSM8K flexible | Drop vs BF16 | C-Eval acc | Drop vs BF16 |
|---|---:|---:|---:|---:|
| BF16 baseline | 0.8802 | - | 0.7905 | - |
| LLM Compressor INT8 W8A8 | 0.8719 | -0.0083 | 0.7853 | -0.0052 |
| LLM Compressor MXFP4A16 | 0.8643 | -0.0159 | 0.7608 | -0.0297 |

The serving benchmark tells a different part of the story: the INT8 W8A8 model gave the strongest online throughput in this setup, while MXFP4A16 had the highest offline decode rate but lower online output throughput.

![vLLM offline decode throughput](assets/figures/vllm_offline_decode.svg)

![vLLM serving output throughput](assets/figures/vllm_serve_output.svg)

| vLLM run | Offline decode tok/s | Serve req/s | Serve output tok/s | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms |
|---|---:|---:|---:|---:|---:|
| BF16 baseline | 2,980 | 2.04 | 261 | 29,250 / 57,082 / 57,981 | 22 / 30 / 31 |
| LLM Compressor INT8 W8A8 | 4,590 | 8.48 | 1,085 | 6,623 / 12,439 / 12,716 | 39 / 39 / 54 |
| LLM Compressor MXFP4A16 | 5,373 | 6.09 | 780 | 7,777 / 17,860 / 18,419 | 81 / 83 / 90 |

The practical takeaway is that INT8 W8A8 is the best-balanced vLLM option in this experiment: it preserves both GSM8K and C-Eval accuracy well, while improving serving output throughput from `261` to `1,085` output tokens/s.

## Activation Diagnosis

The activation probes compare layer hidden states against the BF16 reference. This is useful because the high-level scores alone do not say whether a quantized model is failing everywhere, drifting in a few layers, or accumulating small noise across the stack.

![Activation cosine comparison](assets/figures/activation_diagnosis_cosine_comparison.svg)

![Activation SQNR comparison](assets/figures/activation_diagnosis_sqnr_comparison.svg)

The two INT8 flows behaved very differently:

| Flow | Mean layer cosine | Worst layer | Worst cosine | Worst SQNR | Interpretation |
|---|---:|---|---:|---:|---|
| vLLM LLM Compressor INT8 W8A8 | 0.999886 | layer 35 | 0.998377 | 24.88 dB | Small accumulated quantization noise, no single catastrophic drift point |
| ModelOpt INT8 SmoothQuant | 0.880436 | layer 16 | 0.730791 | 3.25 dB | Clear activation drift around layers 16-21 |

This explains why the vLLM INT8 result remains close to BF16, while the original TensorRT-LLM / ModelOpt INT8 run loses much more accuracy. The layer 16-21 drift was a useful symptom, but later ablations showed that the larger root cause is the W8A8 quantization of `mlp.down_proj`.

## TensorRT-LLM INT8 Root Cause

The first TensorRT-LLM INT8 SmoothQuant result was fast relative to its BF16 engine baseline, but it lost roughly:

- **8.64 points** on GSM8K flexible exact match.
- **9.81 points** on C-Eval accuracy.

Layer activation diagnostics showed severe drift around layers 16-21, but layer-level skip tests did not recover most of the accuracy. Module-level ablations identified `mlp.down_proj` as the important module: keeping all `mlp.down_proj` modules in BF16 recovered C-Eval from `0.6872` to `0.7838`, nearly the TensorRT-LLM BF16 baseline.

The cumulative first-N-layer `mlp.down_proj` ablation makes the trend clearer:

![C-Eval cumulative down_proj](assets/figures/ceval_cumulative_down_proj.svg)

![GSM8K cumulative down_proj](assets/figures/gsm8k_cumulative_down_proj.svg)

Best points:

| Dataset | INT8 baseline | TensorRT-LLM BF16 | Best cumulative skip | Best score |
|---|---:|---:|---|---:|
| C-Eval valid | 0.687221 | 0.785300 | first 34 layers `mlp.down_proj` kept BF16 | 0.787519 |
| GSM8K | 0.827142 | 0.884800 | first 29 layers `mlp.down_proj` kept BF16 | 0.881729 |

The interpretation is mechanical: Qwen3-8B uses a gated MLP. `gate_proj` and `up_proj` expand and transform features, while `down_proj` writes the MLP branch back to the residual stream. Quantization error at that output projection enters the residual path directly and propagates into later layers.

## Documentation

- [Three-scheme quantization comparison](docs/three_quantization_schemes.md)
- [TensorRT-LLM INT8 `mlp.down_proj` root-cause analysis](docs/tensorrt_llm_int8_down_proj_analysis.md)
- [Reproduction guide](docs/reproduction.md)

Some reports are bilingual because the original experiment notes were written in Chinese.

## Repository Layout

```text
assets/figures/          Core charts and explanatory figures
docs/                    Reports and reproduction notes
results/                 Sanitized summary CSV/JSON files only
scripts/
  autoround/             AutoRound INT8 and MXFP4 fakequant workflows
  common/                Shared metric parsing helpers
  tensorrt_llm/          ModelOpt export, TensorRT-LLM build/eval, ablations
  vllm_llmcompressor/    vLLM + LLM Compressor quantization and reports
tests/                   Lightweight parser and diagnosis tests
tools/check_release.py   Release hygiene checks
```

This repository intentionally does **not** include model weights, TensorRT engines, raw sample outputs, large logs, server backups, or private machine paths.

## Quick Start

Install only the lightweight development dependencies:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python tools/check_release.py
```

GPU workflows require separate environment setup for vLLM, LLM Compressor, TensorRT-LLM, ModelOpt, or AutoRound. See the reproduction guide for entry points and expected environment variables.

## License

MIT. See [`LICENSE`](LICENSE).
